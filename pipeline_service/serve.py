from __future__ import annotations

from contextlib import asynccontextmanager
from io import BytesIO
import base64

from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

import pyspz

from config import settings
from logger_config import logger
from schemas import GenerateRequest, GenerateResponse
from modules import GenerationPipeline
from modules.duel_manager import DuelManager
from modules.utils import secure_randint, set_random_seed
from PIL import Image
import io

duel_manager = DuelManager(settings)
pipeline = GenerationPipeline(settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await pipeline.startup()
    # warm up pipeline and vllm
    temp_image = Image.new("RGB", (64, 64), color=(128, 128, 128))
    buffer = io.BytesIO()
    temp_image.save(buffer, format="PNG")
    temp_imge_bytes = buffer.getvalue()
    await run_champion_generation(temp_imge_bytes, -1)
    pipeline._clean_gpu_memory()
    try:
        yield
    finally:
        await pipeline.shutdown()


app = FastAPI(
    title=settings.api_title,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def run_champion_generation(
    image_bytes: bytes, seed: int
) -> tuple[bytes, int, str]:
    final_ply_bytes = None

    if seed < 0:
        seed = secure_randint(0, 10000)
    set_random_seed(seed)

    logger.info("⚡ Preprocessing images (One-time execution)...")
    (
        left_image_without_background,
        right_image_without_background,
        # back_image_without_background,
        original_image_without_background,
    ) = await pipeline.prepare_input_images(image_bytes, seed)

    logger.info("⚔️ ROUND")

    logger.info("   -> Generating Model 1...")
    ply1 = await pipeline.generate_trellis_only(
        [
            left_image_without_background,
            right_image_without_background,
            # back_image_without_background,
        ],
        seed,
        mode="multi_with_voxel_count",
    )

    logger.info("   -> Generating Model 2...")
    ply2 = await pipeline.generate_trellis_only(
        [left_image_without_background, right_image_without_background, original_image_without_background],
        seed,
        mode="multi_sto",
    )

    winner_r1_idx, _ = await duel_manager.run_duel(image_bytes, ply1, ply2)

    if winner_r1_idx == 1:
        final_ply_bytes = ply2
        selected_seed = seed
        logger.success("🏆 TOURNAMENT CHAMPION Model 2")
    else:
        final_ply_bytes = ply1
        selected_seed = seed
        logger.success("🏆 TOURNAMENT CHAMPION Model 1")

    return final_ply_bytes, selected_seed


# ---------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ready"}


@app.post("/generate_from_base64", response_model=GenerateResponse)
async def generate_from_base64(request: GenerateRequest) -> GenerateResponse:
    """
    Endpoint JSON này giữ nguyên, thường dùng cho test single request.
    """
    try:
        # Lưu ý: generate_gs trong pipeline nên được cập nhật để dùng prepare_input_images bên trong
        result = await pipeline.generate_gs(request)

        compressed_ply_bytes = None
        if result.ply_file_base64 and settings.compression:
            compressed_ply_bytes = pyspz.compress(result.ply_file_base64, workers=1)
            logger.info(f"Compressed PLY size: {len(compressed_ply_bytes)} bytes")

        result.ply_file_base64 = base64.b64encode(
            result.ply_file_base64 if not compressed_ply_bytes else compressed_ply_bytes
        ).decode("utf-8")

        return result
    except Exception as exc:
        logger.exception(f"Error generating task: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/generate")
async def generate(
    prompt_image_file: UploadFile = File(...), seed: int = Form(-1)
) -> StreamingResponse:
    """
    Generate PLY (Streaming). Có chạy giải đấu nếu seed = -1.
    """
    try:
        logger.info(
            f"Task received (/generate). Uploading image: {prompt_image_file.filename}"
        )
        image_bytes = await prompt_image_file.read()

        # Gọi hàm chung
        final_ply_bytes, selected_seed = await run_champion_generation(
            image_bytes, seed
        )

        ply_buffer = BytesIO(final_ply_bytes)
        buffer_size = len(ply_buffer.getvalue())
        ply_buffer.seek(0)

        headers = {
            "Content-Length": str(buffer_size),
            "X-Generated-Seed": str(selected_seed),
        }

        async def generate_chunks():
            chunk_size = 1024 * 1024
            while chunk := ply_buffer.read(chunk_size):
                yield chunk

        return StreamingResponse(
            generate_chunks(), media_type="application/octet-stream", headers=headers
        )

    except Exception as exc:
        logger.exception(f"Error generating from upload: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/generate-spz")
async def generate_spz(
    prompt_image_file: UploadFile = File(...), seed: int = Form(-1)
) -> StreamingResponse:
    """
    Generate SPZ (Compressed). BẮT BUỘC CHẠY GIẢI ĐẤU nếu seed = -1 để lấy model tốt nhất.
    """
    try:
        logger.info(
            f"Task received (/generate-spz). Uploading image: {prompt_image_file.filename}"
        )
        image_bytes = await prompt_image_file.read()

        # 1. Gọi hàm chung để lấy model VÔ ĐỊCH (Champion)
        final_ply_bytes, selected_seed = await run_champion_generation(
            image_bytes, seed
        )

        # 2. Nén thành SPZ
        if final_ply_bytes:
            logger.info("Compressing Champion model to SPZ...")
            ply_compressed_bytes = pyspz.compress(final_ply_bytes, workers=1)
            logger.info(f"Task completed. SPZ size: {len(ply_compressed_bytes)} bytes")

            # (Tùy chọn) Có thể trả về Header thông tin người thắng cuộc
            headers = {
                "X-Generated-Seed": str(selected_seed),
            }

            return StreamingResponse(
                BytesIO(ply_compressed_bytes),
                media_type="application/octet-stream",
                headers=headers,
            )
        else:
            raise HTTPException(status_code=500, detail="Generated content is empty")

    except Exception as exc:
        logger.exception(f"Error generating from upload: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/setup/info")
async def get_setup_info() -> dict:
    try:
        return settings.dict()
    except Exception as e:
        logger.error(f"Failed to get setup info: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve configuration")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("serve:app", host=settings.host, port=settings.port, reload=False)
