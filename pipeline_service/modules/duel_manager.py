import io
import base64
import asyncio
import httpx
import torch
import numpy as np
from PIL import Image
from typing import Literal, Tuple, Optional
from pydantic import BaseModel
from openai import AsyncOpenAI
import time
import json
import re
from config import settings
from logger_config import logger

# --- IMPORTS TỪ SOURCE RENDER CỦA BẠN ---
# Đảm bảo bạn đã đặt code renderers đúng chỗ trong project
from .renderers.gs_renderer.renderer import Renderer
from .renderers.ply_loader import PlyLoader


# --- CONSTANTS RENDER ---
IMG_WIDTH = 518
IMG_HEIGHT = 518
GRID_VIEW_GAP = 5
VIEWS_NUMBER = 16
THETA_ANGLES = np.linspace(0, 360, num=VIEWS_NUMBER)
PHI_ANGLES = np.full_like(THETA_ANGLES, -15.0)
GRID_VIEW_INDICES = [1, 5, 9, 13]
CAM_RAD = 2.5
CAM_FOV_DEG = 49.1
REF_BBOX_SIZE = 1.5

# --- CONSTANTS JUDGE ---
SYSTEM_PROMPT = """
You are a specialized 3D model evaluation system. 
Analyze visual quality and prompt adherence with expert precision. 
Always respond with valid JSON only."""
USER_PROMPT_IMAGE = """Does each 3D model match the image prompt?

Penalty 0-10:
0 = Perfect match
3 = Minor issues (slight shape differences, missing small details)
5 = Moderate issues (wrong style, significant details missing)
7 = Major issues (wrong category but related, e.g. chair vs stool)
10 = Completely wrong object

Output: {"penalty_1": <0-10>, "penalty_2": <0-10>, "issues": "<brief>"}"""

class JudgeResponse(BaseModel):
    penalty_1: int
    penalty_2: int
    issues: str

class DuelManager:
    def __init__(self, settings):
        self.settings = settings
        self.client = AsyncOpenAI(
            base_url=settings.vllm_url,
            api_key=settings.vllm_api_key,
            http_client=httpx.AsyncClient(limits=httpx.Limits(max_keepalive_connections=10, max_connections=20))
        )

    def _render_ply_to_grid_bytes(self, ply_bytes: bytes, device: torch.device) -> Optional[bytes]:
        """Render PLY bytes thành PNG bytes (Grid 2x2)"""
        ply_loader = PlyLoader()
        renderer = Renderer()
        
        try:
            # Load PLY từ memory
            gs_data = ply_loader.from_buffer(io.BytesIO(ply_bytes))
            gs_data = gs_data.send_to_device(device)
            
            # Setup Camera angles
            theta_angles = THETA_ANGLES[GRID_VIEW_INDICES].astype("float32")
            phi_angles = PHI_ANGLES[GRID_VIEW_INDICES].astype("float32")
            bg_color = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32).to(device)
            
            # Render 4 views
            images = renderer.render_gs(
                gs_data, views_number=4, img_width=IMG_WIDTH, img_height=IMG_HEIGHT,
                theta_angles=theta_angles, phi_angles=phi_angles,
                cam_rad=CAM_RAD, cam_fov=CAM_FOV_DEG, ref_bbox_size=REF_BBOX_SIZE,
                bg_color=bg_color,
            )
            
            # Combine thành Grid 2x2 (Logic combine_images4)
            row_width = IMG_WIDTH * 2 + GRID_VIEW_GAP
            column_height = IMG_HEIGHT * 2 + GRID_VIEW_GAP
            combined_image = Image.new("RGB", (row_width, column_height), color="black")
            pil_images = [Image.fromarray(img.detach().cpu().numpy()) for img in images]
            
            combined_image.paste(pil_images[0], (0, 0))
            combined_image.paste(pil_images[1], (IMG_WIDTH + GRID_VIEW_GAP, 0))
            combined_image.paste(pil_images[2], (0, IMG_HEIGHT + GRID_VIEW_GAP))
            combined_image.paste(pil_images[3], (IMG_WIDTH + GRID_VIEW_GAP, IMG_HEIGHT + GRID_VIEW_GAP))
            
            buf = io.BytesIO()
            combined_image.save(buf, format="PNG")
            combined_image.save(f"combined_image_{round(time.time(), 2)}.png")
            return buf.getvalue()

        except Exception as e:
            logger.error(f"Render Error: {e}")
            return None
        finally:
            # Dọn dẹp VRAM ngay lập tức
            del ply_loader, renderer, gs_data
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    async def _call_vllm(self, prompt_b64: str, img1_b64: str, img2_b64: str) -> JudgeResponse:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Image prompt to generate 3D model:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{prompt_b64}"}},
                    {"type": "text", "text": "First 3D model (4 different views):"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img1_b64}"}},
                    {"type": "text", "text": "Second 3D model (4 different views):"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img2_b64}"}},
                    {"type": "text", "text": USER_PROMPT_IMAGE},
                ],
            },
        ]

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "judge-response",
                "schema": JudgeResponse.model_json_schema(),
            },
        }

        try:
            start = time.time()
            completion = await self.client.chat.completions.create(
                model=self.settings.vllm_model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=32, 
                response_format=response_format,
            )
            end = time.time()
            logger.info(f"vLLM call time: {end - start} seconds")
            # Log stop reason for debugging
            choice = completion.choices[0]
            finish_reason = choice.finish_reason
            logger.info(f"vLLM finish_reason: {finish_reason}")
            if finish_reason == "length":
                logger.warning("vLLM response was truncated due to max_tokens limit")
            
            # Parse JSON response properly instead of using eval()
            content = choice.message.content
            if not content:
                raise ValueError("Empty response from vLLM")
            content = content.strip()
            logger.info(f"Content: {content}")

            content_preview = completion.choices[0].message.content[:500] if 'completion' in locals() else ''
            penalty_1_match = re.search(r'"penalty_1":\s*(\d+)', content_preview)
            penalty_2_match = re.search(r'"penalty_2":\s*(\d+)', content_preview)
            penalty_1 = int(penalty_1_match.group(1)) if penalty_1_match else 5
            penalty_2 = int(penalty_2_match.group(1)) if penalty_2_match else 5
            issues = ""
            if finish_reason != "length":
                try:
                    issues = json.loads(content)['issues']
                    logger.info(f"Issues: {issues}")
                except json.JSONDecodeError:
                    issues = "Incomplete JSON"
            return JudgeResponse(penalty_1=penalty_1, penalty_2=penalty_2, issues=issues)
            
        except Exception as e:
            logger.error(f"vLLM Call Failed: {e}")
            if 'completion' in locals():
                logger.error(f"Response content: {completion.choices[0].message.content[:500]}")
            return JudgeResponse(penalty_1=5, penalty_2=5, issues="JSON Parse Error")

    async def run_duel(self, prompt_bytes: bytes, ply1_bytes: bytes, ply2_bytes: bytes) -> Tuple[int, str]:
        """
        Trả về (Winner_Index, Issues). 
        Winner_Index: -1 (Seed1 wins), 1 (Seed2 wins), 0 (Draw)
        """
        # save prompt_bytes to file
        temp_image = Image.open(io.BytesIO(prompt_bytes))
        temp_image.save(f"prompt_{int(time.time())}.png")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loop = asyncio.get_running_loop()

        # 1. Render PLY to Images (Chạy trong thread pool để không block async loop)
        logger.info("Rendering PLY files for Duel...")
        img1_bytes = await loop.run_in_executor(None, self._render_ply_to_grid_bytes, ply1_bytes, device)
        img2_bytes = await loop.run_in_executor(None, self._render_ply_to_grid_bytes, ply2_bytes, device)

        if not img1_bytes or not img2_bytes:
            logger.error("Render failed, cannot judge.")
            return 0, "Render Failure"

        # 2. Prepare Base64
        prompt_b64 = base64.b64encode(prompt_bytes).decode('utf-8').strip()
        render1_b64 = base64.b64encode(img1_bytes).decode('utf-8').strip()
        render2_b64 = base64.b64encode(img2_bytes).decode('utf-8').strip()

        # 3. Position-Balanced Duel (Đấu 2 lượt đảo vị trí)
        logger.info("Asking Judge (vLLM)...")
        res_direct, res_swapped = await asyncio.gather(
            self._call_vllm(prompt_b64, render1_b64, render2_b64),
            self._call_vllm(prompt_b64, render2_b64, render1_b64)
        )

        # Tính toán điểm (Direct: P1 vs P2 | Swapped: P2 vs P1)
        # Seed 1 là P1 ở lượt 1, và là P2 ở lượt 2
        # Handle both dict and JudgeResponse objects
        if isinstance(res_direct, JudgeResponse):
            score1 = (res_direct.penalty_1 + res_swapped.penalty_2) / 2
            logger.info(f"Score 1: direct {res_direct.penalty_1} swapped {res_swapped.penalty_2} score {score1}")
            score2 = (res_swapped.penalty_1 + res_direct.penalty_2) / 2
            logger.info(f"Score 2: direct {res_swapped.penalty_1} swapped {res_direct.penalty_2} score {score2}")
            issues = f"Direct: {res_direct.issues} | Swapped: {res_swapped.issues}"
        else:
            score1 = (res_direct.get('penalty_1', 0) + res_swapped.get('penalty_2', 0)) / 2
            logger.info(f"Score 1: direct {res_direct.get('penalty_1', 0)} swapped {res_swapped.get('penalty_2', 0)} score {score1}")
            score2 = (res_swapped.get('penalty_1', 0) + res_direct.get('penalty_2', 0)) / 2
            logger.info(f"Score 2: direct {res_swapped.get('penalty_1', 0)} swapped {res_direct.get('penalty_2', 0)} score {score2}")
            issues = f"Direct: {res_direct.get('issues', '')} | Swapped: {res_swapped.get('issues', '')}"

        if score1 < score2:
            return -1, issues # Seed 1 Wins (Penalty thấp hơn)
        else:
            return 1, issues # Seed 2 Wins