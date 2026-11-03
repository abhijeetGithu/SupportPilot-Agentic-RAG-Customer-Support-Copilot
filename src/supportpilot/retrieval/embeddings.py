"""Dense (bge-small) and sparse (BM25) encoders via fastembed. CPU-only, ONNX, no torch."""

from functools import lru_cache

from fastembed import SparseEmbedding, SparseTextEmbedding, TextEmbedding

from supportpilot.config import get_settings


def _low_memory_sessions() -> None:
    """fastembed hardcodes ORT_ENABLE_ALL, which keeps fused copies of the weights in memory. It only lets
    callers set enable_cpu_mem_arena, through this hook, so wrap the hook to lower the optimization level too."""
    import onnxruntime as ort
    from fastembed.common.onnx_model import OnnxModel

    original = OnnxModel.add_extra_session_options.__func__
    if getattr(original, "low_memory", False):
        return

    def add_extra_session_options(cls, session_options, extra_options):
        original(cls, session_options, extra_options)
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        session_options.enable_mem_pattern = False

    add_extra_session_options.low_memory = True
    OnnxModel.add_extra_session_options = classmethod(add_extra_session_options)


@lru_cache
def dense_model() -> TextEmbedding:
    s = get_settings()
    if s.onnx_low_memory:
        _low_memory_sessions()
    return TextEmbedding(s.embedding_model, cache_dir=s.embedding_cache_dir, threads=1,
                         enable_cpu_mem_arena=False)  # memory over speed: 512 MB on the free tier


@lru_cache
def sparse_model() -> SparseTextEmbedding:
    s = get_settings()
    return SparseTextEmbedding(s.sparse_model, cache_dir=s.embedding_cache_dir)


def dense_dim() -> int:
    return len(next(dense_model().query_embed("dim")))


def embed_passages(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    return [v.tolist() for v in dense_model().passage_embed(texts, batch_size=batch_size)]


def embed_query(query: str) -> list[float]:
    return next(dense_model().query_embed(query)).tolist()


def sparse_passages(texts: list[str], batch_size: int = 64) -> list[SparseEmbedding]:
    return list(sparse_model().passage_embed(texts, batch_size=batch_size))


def sparse_query(query: str) -> SparseEmbedding:
    return next(sparse_model().query_embed(query))
