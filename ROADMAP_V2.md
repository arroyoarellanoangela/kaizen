# Kaizen V2 Roadmap

> **V2 is gated by evidence, not by ambition.**
> Primero limpiar la base. Luego abstraer la arquitectura. Añadir complejidad solo cuando el corpus o las métricas lo exijan.

---

## Principio rector

V1 demostró que las decisiones de optimización deben estar respaldadas por benchmarks, no por intuición.
V2 aplica el mismo principio a las decisiones de arquitectura:

- Ninguna fase arranca sin un trigger medible o un criterio de entrada explícito.
- La arquitectura crece para soportar evidencia real, no para anticipar casos hipotéticos.
- Multi-index, multi-model y LLM routing son capacidades posibles, no objetivos inevitables.

---

## Pre-V2 — Cleanup de V1

Trabajo previo antes de arrancar V2. No es V2; es la base que lo hace viable.

### 1. Config centralizada

**Problema:** valores hardcoded dispersos en `retriever.py`, `api.py` y `app.py`.

**Acción:** mover a `rag/config.py`:

| Valor | Ubicación actual |
|---|---|
| `qwen3:8b` (LLM model) | `api.py:89`, `app.py:207` |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` (reranker) | `retriever.py:22` |
| `OVERFETCH_FACTOR = 4` | `retriever.py:23` |
| `EMBED_BATCH = 256` | `store.py:55` |
| `ADD_BATCH_SIZE = 500` | `store.py:54` |
| `batch_size=32` (reranker predict) | `retriever.py:94` |
| Workers = 8 (ThreadPoolExecutor) | `app.py:137`, `api.py:215`, `ingest.py:46` |

### 2. Deduplicación de helpers

**Problema:** lógica duplicada entre `app.py` y `api.py`.

**Acción:** extraer a módulos compartidos:

- `rag/monitoring.py` ← `_gpu_metrics()` (duplicada en ambos archivos)
- `rag/pipeline.py` ← `_read_and_chunk()` (duplicada en ambos archivos)
- `rag/prompts.py` ← `SYSTEM_PROMPT` y constantes de generación (duplicadas)

### 3. Logging estructurado

**Problema:** todo usa `print()`. Sin trazabilidad, sin niveles, sin formato.

**Acción:**
- Sustituir todos los `print()` por `logging.getLogger(__name__)`
- Configurar un handler base en `rag/config.py` o `rag/__init__.py`
- Nivel por defecto: `INFO`. Errores silenciados actualmente → `WARNING` o `ERROR`

**Criterio de cierre:** cero `print()` en módulos `rag/` y en `api.py`.

---

## V2.1 — Orchestration Foundation

**Objetivo:** desacoplar el pipeline de sus implementaciones concretas. Que sea posible cambiar de modelo, índice o LLM sin tocar el flujo de query.

### Módulos nuevos

#### `rag/model_registry.py`

Registro central de modelos de embedding y reranker.

```python
# Interfaz mínima
def get_embed_model(name: str = "default") -> SentenceTransformer: ...
def get_reranker(name: str = "default") -> CrossEncoder: ...
def list_models() -> dict[str, dict]: ...  # nombre → {type, model_id, precision}
```

Internamente sigue usando los mismos modelos de V1. El valor es la interfaz, no el modelo.

#### `rag/index_registry.py`

Registro central de colecciones ChromaDB.

```python
# Interfaz mínima
def get_index(name: str = "default") -> chromadb.Collection: ...
def list_indexes() -> list[str]: ...
def route_to_index(query: str, hint: str | None = None) -> str: ...
```

En V2.1, `route_to_index` devuelve siempre `"default"`. La interfaz ya existe para V2.2.

#### `rag/orchestrator.py`

Capa de decisión. Recibe una query y devuelve un plan de ejecución.

```python
@dataclass
class RoutePlan:
    indexes: list[str]        # qué colecciones consultar
    embed_model: str          # qué modelo usar para embeddings
    use_reranker: bool        # si aplicar cross-encoder
    reranker_model: str       # qué reranker usar
    llm_model: str            # qué LLM usar para generación
    mode: str                 # "answer" | "summary" | "code"
    reason: str               # por qué se eligió esta ruta (trazabilidad)

def plan(query: str, category: str | None = None) -> RoutePlan: ...
```

El routing en V2.1 es **determinista y explícito**: reglas simples, sin IA dentro del router.

Ejemplo de reglas iniciales:
- `mode = "code"` si la query contiene palabras clave de código (función, clase, `def`, `import`, etc.)
- `mode = "summary"` si la query supera 150 caracteres o pide síntesis
- `mode = "answer"` en todos los demás casos
- `indexes = ["default"]` siempre (en V2.1)
- `use_reranker = True` siempre (en V2.1)

### Tracing de decisiones

Cada query debe dejar un log del plan elegido:

```
[orchestrator] query="..." → indexes=["default"] model="all-MiniLM-L6-v2" reranker=True llm="qwen3:8b" mode="answer" reason="default path"
```

Esto no es opcional. Sin trazabilidad, el routing es una caja negra inútil.

### Criterio de cierre de V2.1

- `api.py` y `app.py` llaman a `orchestrator.plan()`, no directamente a `search()` ni a `get_collection()`
- Todo cambio de modelo o índice se hace en los registros, no en el código de query
- Cada query genera un log de decisión verificable

---

## V2.2 — Second Index

**Objetivo:** soporte real para un segundo corpus con características distintas al actual.

### Trigger de entrada

**No arrancar sin al menos uno de estos:**

1. Degradación documentada: NDCG@5 < 0.75 en un subconjunto identificable del corpus (ej. PDFs longform)
2. Corpus nuevo que no comparte dominio semántico con el actual (ej. código fuente, normativa, libros técnicos)
3. Evidencia de contaminación: queries de un dominio recuperando chunks irrelevantes de otro con frecuencia > 20%

### Qué entra

- Nuevo índice ChromaDB con su propia configuración (chunk size, metadata, embedding)
- `index_registry.py` actualizado con routing real por tipo de documento
- `route_to_index()` con lógica basada en categoría, extensión o señales del corpus
- Validación de calidad del nuevo índice antes de activarlo en producción (NDCG@5 en subset representativo)

### Qué no entra aún

- Segundo modelo de embedding — el mismo `all-MiniLM-L6-v2` sirve para los dos índices inicialmente
- Merge de resultados cross-index — se elige un índice por query, no se fusionan

### Criterio de cierre de V2.2

- Dos índices activos en `index_registry`
- Routing documentado y trazado en logs
- NDCG@5 del nuevo índice validado y publicado (mismo formato que `RERANKER_QUALITY.md`)

---

## V2.3 — Second Embedding Model

**Objetivo:** validar si un segundo modelo de embedding mejora la calidad en algún subconjunto del corpus.

### Trigger de entrada

**No arrancar sin:**

1. V2.2 completada y estable
2. Evidencia de que `all-MiniLM-L6-v2` es el cuello de botella de calidad en al menos un índice:
   - NDCG@5 plateau en ese índice a pesar de mejoras en chunking o reranker
   - O un modelo candidato con benchmark público que muestre >5% de mejora en el dominio específico

### Qué entra

- Modelo candidato evaluado contra `all-MiniLM-L6-v2` en el corpus real (no en benchmarks externos)
- `model_registry.py` actualizado con el nuevo modelo
- Benchmark de throughput comparativo (FP16 vs FP32, latencia, VRAM)
- Validación de calidad: NDCG@5/10 antes vs después
- Decisión explícita de qué índice usa qué modelo (no por defecto "todos usan el nuevo")

### Criterio de cierre de V2.3

- Dos modelos en `model_registry` con asignación explícita por índice
- Benchmark y validación de calidad publicados
- Mejora de NDCG@5 ≥ 3% sobre el subconjunto objetivo para justificar la complejidad añadida

---

## V2.4 — LLM Routing

**Objetivo:** enrutar queries a distintos LLMs según el tipo de tarea.

### Trigger de entrada

**No arrancar sin:**

1. V2.1 completada (el orchestrator ya existe como base)
2. Al menos dos casos de uso reales y distintos que justifiquen LLMs distintos:
   - Ejemplo: respuestas rápidas de FAQ vs razonamiento complejo vs edición de código
3. Dos modelos Ollama locales disponibles y benchmarkeados en latencia y calidad

### Diseño

El routing sigue siendo **determinista** en V2.4:

```python
# Ejemplo de reglas de routing LLM
if plan.mode == "code":
    plan.llm_model = "codellama:7b"  # o el modelo de código disponible
elif len(query) > 200 or "compara" in query or "sintetiza" in query:
    plan.llm_model = "qwen3:14b"     # modelo más potente para razonamiento
else:
    plan.llm_model = "qwen3:8b"      # default: velocidad
```

No se introduce un meta-LLM para decidir qué LLM usar. Las reglas son código, no inferencia.

### Criterio de cierre de V2.4

- Al menos dos rutas de LLM activas con triggers documentados
- Latencia de cada ruta medida y publicada
- Logs de routing verificables por query

---

## Out of Scope para V2

Las siguientes capacidades **no entran en ninguna fase de V2** sin una revisión explícita:

| Capacidad | Razón |
|---|---|
| Agentic routing libre (LLM decide el plan) | Opaco, difícil de debuggear, no hay evidencia de que aporte vs reglas |
| Tres o más índices simultáneos | Sin corpus que lo justifique, añade complejidad sin ganancia clara |
| Merge de resultados cross-index | Fusionar rankings de índices distintos requiere calibración que no tenemos |
| Tool use / function calling | Distinto paradigma, fuera del scope de retrieval engine |
| Code index y code editing | Requiere corpus específico y evaluación dedicada |
| Autenticación y multi-usuario | Infraestructura, no motor. No es el foco de este proyecto |
| Deployment en producción (Docker, SSL) | Fuera del alcance de un portfolio engine |

---

## Estado actual

| Fase | Estado | Condición de entrada |
|---|---|---|
| Pre-V2 cleanup | ✅ Completado | — |
| Pre-V2.5 Generation polish | ✅ Completado | SYSTEM_PROMPT + upgrade 8B→14B |
| LLM Provider abstraction | ✅ Completado | `rag/llm.py` — Ollama + OpenAI-compatible APIs |
| V2.1 Orchestration | ✅ Completado | `model_registry` + `index_registry` + `orchestrator` |
| V2.2 Second Index | Pendiente | V2.1 + trigger de corpus |
| V2.3 Second Embedding Model | Pendiente | V2.2 + plateau de calidad documentado |
| V2.4 LLM Routing | Pendiente | V2.1 ✅ + dos casos de uso reales |
