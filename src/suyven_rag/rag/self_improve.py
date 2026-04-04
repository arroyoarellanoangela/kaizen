"""Self-Improvement Agent — Investigates GitHub repos, implements and tests improvements.

Full autonomous loop:
  1. ANALYZE: Fetch GitHub repos, extract patterns with LLM
  2. PLAN: Prioritize improvements by impact/effort
  3. IMPLEMENT: Generate code in isolated sandbox (git worktree)
  4. TEST: Run test suite against sandbox
  5. EVALUATE: Run eval_suite to measure actual impact
  6. DECIDE: If metrics improve -> merge. If not -> revert. If obsolete -> delete.

Architecture:
  GitHubScout -> PatternExtractor -> ImprovementPlanner -> Implementer -> Sandbox -> Evaluator -> Merger

Usage:
    # Analyze repos
    python -m rag.self_improve --repo https://github.com/unslothai/unsloth

    # Search and analyze
    python -m rag.self_improve --discover "embedding fine-tuning optimization"

    # Show improvement plan
    python -m rag.self_improve --plan

    # Run full auto-improve loop (analyze -> implement -> test -> merge/revert)
    python -m rag.self_improve --auto

    # Implement a specific improvement by ID
    python -m rag.self_improve --implement 20260315_01
"""

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
IMPROVEMENTS_DIR = BASE_DIR / "data" / "self_improve"
KNOWLEDGE_FILE = IMPROVEMENTS_DIR / "knowledge_base.json"
SANDBOX_DIR = BASE_DIR / "data" / "self_improve" / "sandbox"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class RepoAnalysis:
    """Analysis of a single GitHub repo."""

    url: str
    name: str
    description: str = ""
    stars: int = 0
    language: str = ""
    readme_summary: str = ""
    key_patterns: list[str] = field(default_factory=list)
    relevant_files: list[str] = field(default_factory=list)
    applicable_improvements: list[dict] = field(default_factory=list)
    analyzed_at: str = ""


@dataclass
class Improvement:
    """A proposed improvement to Suyven."""

    id: str
    title: str
    description: str
    source_repo: str
    category: str  # training | eval | data | architecture | optimization
    priority: str  # high | medium | low
    effort: str  # small | medium | large
    target_files: list[str] = field(default_factory=list)
    code_snippet: str = ""
    status: str = "proposed"  # proposed | implementing | testing | passed | failed | merged | rejected | obsolete
    created_at: str = ""
    implemented_at: str = ""
    test_results: dict = field(default_factory=dict)
    eval_before: dict = field(default_factory=dict)
    eval_after: dict = field(default_factory=dict)
    rejection_reason: str = ""


# ---------------------------------------------------------------------------
# GitHub Scout
# ---------------------------------------------------------------------------


class GitHubScout:
    """Fetches GitHub repo metadata and content."""

    GITHUB_API = "https://api.github.com"

    def __init__(self, token: str | None = None):
        self.session = requests.Session()
        self.session.headers["Accept"] = "application/vnd.github.v3+json"
        self.session.headers["User-Agent"] = "Suyven-SelfImprove/1.0"
        if token:
            self.session.headers["Authorization"] = f"token {token}"

    def _parse_repo(self, url: str) -> tuple[str, str]:
        url = url.rstrip("/").rstrip(".git")
        parts = urlparse(url)
        segments = parts.path.strip("/").split("/")
        if len(segments) >= 2:
            return segments[0], segments[1]
        raise ValueError(f"Cannot parse GitHub URL: {url}")

    def get_repo_info(self, url: str) -> dict:
        owner, repo = self._parse_repo(url)
        resp = self.session.get(f"{self.GITHUB_API}/repos/{owner}/{repo}", timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            return {
                "name": data.get("full_name", ""),
                "description": data.get("description", ""),
                "stars": data.get("stargazers_count", 0),
                "language": data.get("language", ""),
                "topics": data.get("topics", []),
                "default_branch": data.get("default_branch", "main"),
            }
        logger.warning("GitHub API returned %d for %s", resp.status_code, url)
        return {"name": f"{owner}/{repo}", "description": "", "stars": 0}

    def get_readme(self, url: str) -> str:
        owner, repo = self._parse_repo(url)
        resp = self.session.get(
            f"{self.GITHUB_API}/repos/{owner}/{repo}/readme",
            headers={"Accept": "application/vnd.github.v3.raw"},
            timeout=15,
        )
        return resp.text[:15000] if resp.status_code == 200 else ""

    def get_tree(self, url: str, max_depth: int = 2) -> list[str]:
        owner, repo = self._parse_repo(url)
        info = self.get_repo_info(url)
        branch = info.get("default_branch", "main")
        resp = self.session.get(
            f"{self.GITHUB_API}/repos/{owner}/{repo}/git/trees/{branch}?recursive=1",
            timeout=15,
        )
        if resp.status_code == 200:
            tree = resp.json().get("tree", [])
            return [
                item["path"]
                for item in tree
                if item["type"] == "blob" and item["path"].count("/") < max_depth
            ][:200]
        return []

    def get_file(self, url: str, path: str) -> str:
        owner, repo = self._parse_repo(url)
        resp = self.session.get(
            f"{self.GITHUB_API}/repos/{owner}/{repo}/contents/{path}",
            headers={"Accept": "application/vnd.github.v3.raw"},
            timeout=15,
        )
        return resp.text[:10000] if resp.status_code == 200 else ""

    def search_repos(self, query: str, max_results: int = 10) -> list[dict]:
        resp = self.session.get(
            f"{self.GITHUB_API}/search/repositories",
            params={"q": query, "sort": "stars", "order": "desc", "per_page": max_results},
            timeout=15,
        )
        if resp.status_code == 200:
            return [
                {
                    "url": item["html_url"],
                    "name": item["full_name"],
                    "description": item.get("description", ""),
                    "stars": item.get("stargazers_count", 0),
                }
                for item in resp.json().get("items", [])
            ]
        return []


# ---------------------------------------------------------------------------
# Pattern Extractor
# ---------------------------------------------------------------------------


class PatternExtractor:
    """Uses LLM (with multi-backend fallback) to extract actionable patterns."""

    SYSTEM_PROMPT = """You are an expert ML engineer analyzing GitHub repositories.
Your job is to extract specific, actionable technical patterns that could improve
a RAG (Retrieval-Augmented Generation) engine with these components:
- Embedding model: BAAI/bge-m3 (568M params, 1024-dim) on GPU FP16
- ChromaDB vector store (33K+ chunks)
- Cross-encoder reranker (ms-marco-MiniLM-L-6-v2)
- FastAPI backend + React frontend
- GPU: RTX 5070, Python 3.12

For each improvement, include the EXACT code changes needed — which functions to modify,
what to add, what to replace. Be surgical and specific."""

    EXTRACT_PROMPT = """Analyze this GitHub repo and extract actionable improvements for our RAG system.

Repo: {repo_name} ({stars} stars)
Description: {description}

README (excerpt):
{readme}

File structure:
{tree}
{extra_files}

Return a JSON array of improvements:
[{{"title": "...", "description": "...", "category": "training|eval|data|architecture|optimization",
   "priority": "high|medium|low", "effort": "small|medium|large",
   "target_files": ["finetune/train.py"], "code_hint": "pseudocode or key implementation"}}]

Return ONLY the JSON array, no markdown."""

    IMPLEMENT_PROMPT = """You are implementing an improvement for a RAG fine-tuning system.

IMPROVEMENT:
  Title: {title}
  Description: {description}
  Code hint: {code_hint}

EXISTING CODE in {target_file} (DO NOT REWRITE THIS — only add new code):
```python
{existing_code}
```

RELATED FILES context:
{context}

CRITICAL RULES:
1. Generate ONLY the NEW imports and NEW functions/classes to ADD
2. Do NOT rewrite or repeat any existing code
3. The output will be APPENDED to the end of the existing file
4. Any new imports must go at the top of your output
5. New functions must have unique names that don't conflict with existing ones
6. Include proper docstrings and type hints
7. Use logging (import logging; logger = logging.getLogger(__name__))

Return ONLY the new Python code to append. No markdown fences, no explanation.
Start with any needed imports, then the new functions."""

    # LLM backends by role:
    #   "analysis" = cheap/fast for repo analysis + JSON extraction
    #   "code"     = top-tier for code generation (needs to compile!)
    ANALYSIS_BACKENDS = [
        {
            "name": "groq",
            "url_env": "LLM_API_URL",
            "key_env": "LLM_API_KEY",
            "model_env": "LLM_MODEL",
        },
        {"name": "ollama", "url": "http://localhost:11434/v1", "key": "", "model": "qwen3:14b"},
    ]
    CODE_BACKENDS = [
        {
            "name": "gemini",
            "url": "https://generativelanguage.googleapis.com/v1beta/openai",
            "key_env": "GEMINI_API_KEY",
            "model": "gemini-2.5-flash",
        },
        {
            "name": "groq",
            "url_env": "LLM_API_URL",
            "key_env": "LLM_API_KEY",
            "model_env": "LLM_MODEL",
        },
        {"name": "ollama", "url": "http://localhost:11434/v1", "key": "", "model": "qwen3:14b"},
    ]

    def __init__(self, preferred_backend: str | None = None):
        from .config import LLM_API_KEY, LLM_API_URL, LLM_MODEL

        gemini_key = os.environ.get("GEMINI_API_KEY", "")

        def _build_backends(backend_defs):
            backends = []
            for b in backend_defs:
                if preferred_backend and b["name"] != preferred_backend:
                    continue
                entry = {"name": b["name"]}
                if "url_env" in b:
                    entry["url"] = LLM_API_URL
                    entry["key"] = LLM_API_KEY
                    entry["model"] = LLM_MODEL
                elif b.get("key_env") == "GEMINI_API_KEY":
                    if not gemini_key:
                        continue  # Skip if no Gemini key
                    entry["url"] = b["url"]
                    entry["key"] = gemini_key
                    entry["model"] = b["model"]
                else:
                    entry["url"] = b["url"]
                    entry["key"] = b.get("key", "")
                    entry["model"] = b["model"]
                backends.append(entry)
            return backends

        self.backends = _build_backends(self.ANALYSIS_BACKENDS)  # For analysis
        self.code_backends = _build_backends(self.CODE_BACKENDS)  # For code gen

    def _call_llm(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 4000,
        temperature: float = 0.3,
        role: str = "analysis",
    ) -> str | None:
        """Call LLM with automatic backend fallback.

        Args:
            role: "analysis" for repo analysis, "code" for code generation.
                  Uses different backend lists depending on role.
        """
        sys_msg = system or self.SYSTEM_PROMPT
        backend_list = self.code_backends if role == "code" else self.backends

        for backend in backend_list:
            headers = {"Content-Type": "application/json"}
            if backend["key"]:
                headers["Authorization"] = f"Bearer {backend['key']}"

            try:
                resp = requests.post(
                    f"{backend['url']}/chat/completions",
                    headers=headers,
                    json={
                        "model": backend["model"],
                        "messages": [
                            {"role": "system", "content": sys_msg},
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    },
                    timeout=180,
                )
                if resp.status_code == 429:
                    logger.warning("Rate limited on %s, trying next backend", backend["name"])
                    continue
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                # Strip thinking tags (qwen3/deepseek)
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
                return content.strip()

            except Exception as e:
                logger.warning("LLM call failed (%s): %s", backend["name"], e)
                continue

        return None

    def _parse_json(self, content: str) -> list[dict]:
        """Parse JSON array from LLM response."""
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```\w*\n?", "", content)
            content = re.sub(r"\n?```$", "", content)
            content = content.strip()

        try:
            result = json.loads(content)
            if isinstance(result, list):
                return result
            if isinstance(result, dict):
                for key in ["improvements", "patterns", "results", "data"]:
                    if key in result:
                        return result[key]
                return [result]
        except json.JSONDecodeError:
            pass

        match = re.search(r"\[.*\]", content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        logger.warning("Could not parse JSON: %s...", content[:200])
        return []

    def extract_offline(
        self, repo_info: dict, readme: str, tree: list[str], extra_content: str = ""
    ) -> list[dict]:
        """Offline pattern extraction using regex (no LLM needed)."""
        all_text = (readme + "\n" + extra_content).lower()
        found = []

        PATTERNS = [
            (
                r"flash.?attention|flash_attn",
                {
                    "title": "Flash Attention for faster training",
                    "description": "Use Flash Attention for O(N) memory attention computation during LoRA training",
                    "category": "optimization",
                    "priority": "high",
                    "effort": "medium",
                    "target_files": ["finetune/train.py", "finetune/optimizations.py"],
                    "code_hint": "from flash_attn import flash_attn_func",
                },
            ),
            (
                r"gradient.?checkpoint|activation.?checkpoint",
                {
                    "title": "Gradient checkpointing for memory savings",
                    "description": "Trade ~25% speed for ~40% VRAM reduction during backprop",
                    "category": "optimization",
                    "priority": "medium",
                    "effort": "small",
                    "target_files": ["finetune/train.py", "finetune/optimizations.py"],
                    "code_hint": "model.gradient_checkpointing_enable()",
                },
            ),
            (
                r"bf16|bfloat16|mixed.?precision",
                {
                    "title": "BFloat16 training (Ampere+ GPUs)",
                    "description": "BF16 is more stable than FP16 for training, no GradScaler needed",
                    "category": "optimization",
                    "priority": "high",
                    "effort": "small",
                    "target_files": ["finetune/train.py", "finetune/optimizations.py"],
                    "code_hint": "torch.amp.autocast('cuda', dtype=torch.bfloat16)",
                },
            ),
            (
                r"qlora|4.?bit|quantiz",
                {
                    "title": "QLoRA: 4-bit quantized base + LoRA adapters",
                    "description": "Quantize base model to 4-bit, train only LoRA in FP16 -- 75% VRAM reduction",
                    "category": "training",
                    "priority": "high",
                    "effort": "medium",
                    "target_files": ["finetune/lora.py", "finetune/train.py"],
                    "code_hint": "BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)",
                },
            ),
            (
                r"dpo|direct.?preference|preference.?optim",
                {
                    "title": "DPO training for embedding preference learning",
                    "description": "Train embeddings to prefer relevant passages over irrelevant ones directly",
                    "category": "training",
                    "priority": "high",
                    "effort": "large",
                    "target_files": ["finetune/train.py", "finetune/losses.py"],
                    "code_hint": "DPO loss: log_sigmoid(beta * (score_chosen - score_rejected))",
                },
            ),
            (
                r"matryoshka|adaptive.?dim",
                {
                    "title": "Matryoshka embeddings (adaptive dimensionality)",
                    "description": "Train embeddings that work at multiple dimensions (384, 256, 128, 64)",
                    "category": "training",
                    "priority": "medium",
                    "effort": "medium",
                    "target_files": ["finetune/train.py", "finetune/losses.py"],
                    "code_hint": "Loss at multiple truncated dimensions: sum(loss(emb[:d]) for d in dims)",
                },
            ),
            (
                r"hard.?negativ|contrastive.*min",
                {
                    "title": "Hard negative mining during training",
                    "description": "Mine hard negatives from in-batch embeddings for stronger contrastive signal",
                    "category": "data",
                    "priority": "high",
                    "effort": "medium",
                    "target_files": ["finetune/train.py", "finetune/data_gen_v2.py"],
                    "code_hint": "Select negatives with highest similarity that are still incorrect",
                },
            ),
            (
                r"early.?stop|patience|best.?model",
                {
                    "title": "Early stopping with best model checkpoint",
                    "description": "Stop training when eval loss plateaus, keep best checkpoint",
                    "category": "training",
                    "priority": "medium",
                    "effort": "small",
                    "target_files": ["finetune/train.py"],
                    "code_hint": "if eval_loss < best_loss: save_checkpoint(); patience_counter = 0",
                },
            ),
            (
                r"onnx|tensorrt|openvino",
                {
                    "title": "ONNX/TensorRT export for faster inference",
                    "description": "Export fine-tuned model to ONNX for 2-4x faster inference",
                    "category": "optimization",
                    "priority": "medium",
                    "effort": "medium",
                    "target_files": ["finetune/export.py", "rag/model_registry.py"],
                    "code_hint": "torch.onnx.export(model, dummy_input, 'model.onnx')",
                },
            ),
            (
                r"packing|sequence.?pack",
                {
                    "title": "Sequence packing (eliminate padding waste)",
                    "description": "Pack multiple short sequences into single batch slots -- 30% faster",
                    "category": "optimization",
                    "priority": "high",
                    "effort": "medium",
                    "target_files": ["finetune/optimizations.py", "finetune/train.py"],
                    "code_hint": "Pack sequences to fill max_length slots, use attention mask for boundaries",
                },
            ),
        ]

        seen = set()
        for pattern, improvement in PATTERNS:
            if re.search(pattern, all_text) and improvement["title"] not in seen:
                seen.add(improvement["title"])
                found.append(improvement.copy())

        return found

    def extract(
        self, repo_info: dict, readme: str, tree: list[str], extra_content: str = ""
    ) -> list[dict]:
        """Extract improvements -- tries LLM backends, falls back to offline."""
        tree_str = "\n".join(tree[:100])
        extra = f"\nKey file contents:\n{extra_content}" if extra_content else ""

        prompt = self.EXTRACT_PROMPT.format(
            repo_name=repo_info.get("name", ""),
            description=repo_info.get("description", ""),
            stars=repo_info.get("stars", 0),
            readme=readme[:8000],
            tree=tree_str,
            extra_files=extra[:5000],
        )

        content = self._call_llm(prompt)
        if content:
            results = self._parse_json(content)
            if results:
                return results

        logger.info("LLM extraction failed, using offline extraction")
        return self.extract_offline(repo_info, readme, tree, extra_content)

    def generate_implementation(
        self, improvement: dict, target_file: Path, context_files: list[Path] | None = None
    ) -> str | None:
        """Generate code implementation for an improvement."""
        existing_code = ""
        if target_file.exists():
            existing_code = target_file.read_text(encoding="utf-8")

        context = ""
        for cf in context_files or []:
            if cf.exists():
                context += f"\n--- {cf.name} ---\n{cf.read_text(encoding='utf-8')[:3000]}\n"

        prompt = self.IMPLEMENT_PROMPT.format(
            title=improvement.get("title", ""),
            description=improvement.get("description", ""),
            code_hint=improvement.get("code_snippet") or improvement.get("code_hint", ""),
            target_file=target_file.name,
            existing_code=existing_code[:8000],
            context=context[:5000],
        )

        system = """You are a senior Python engineer. Generate clean, production-ready code.
Return ONLY the complete Python file content. No markdown, no explanation."""

        return self._call_llm(prompt, system=system, max_tokens=8000, temperature=0.2, role="code")


# ---------------------------------------------------------------------------
# Knowledge Base
# ---------------------------------------------------------------------------


class KnowledgeBase:
    """Persistent storage for repo analyses and improvements."""

    def __init__(self, path: Path = KNOWLEDGE_FILE):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                return json.load(f)
        return {"repos": {}, "improvements": [], "last_updated": ""}

    def save(self) -> None:
        self.data["last_updated"] = datetime.now().isoformat()
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def add_repo(self, analysis: RepoAnalysis) -> None:
        self.data["repos"][analysis.url] = asdict(analysis)

    def add_improvements(self, improvements: list[Improvement]) -> None:
        existing_titles = {i["title"] for i in self.data["improvements"]}
        for imp in improvements:
            if imp.title not in existing_titles:
                self.data["improvements"].append(asdict(imp))

    def get_improvements(
        self, status: str | None = None, category: str | None = None
    ) -> list[dict]:
        results = self.data["improvements"]
        if status:
            results = [i for i in results if i["status"] == status]
        if category:
            results = [i for i in results if i["category"] == category]
        return results

    def update_improvement(self, imp_id: str, **updates) -> None:
        for imp in self.data["improvements"]:
            if imp["id"] == imp_id:
                imp.update(updates)
                break
        self.save()

    def mark_obsolete(self, imp_id: str, reason: str) -> None:
        self.update_improvement(imp_id, status="obsolete", rejection_reason=reason)

    def get_stats(self) -> dict:
        improvements = self.data["improvements"]
        return {
            "total_repos": len(self.data["repos"]),
            "total_improvements": len(improvements),
            "by_status": {
                s: len([i for i in improvements if i["status"] == s])
                for s in sorted(set(i["status"] for i in improvements))
            }
            if improvements
            else {},
            "by_category": {
                c: len([i for i in improvements if i["category"] == c])
                for c in sorted(set(i["category"] for i in improvements))
            }
            if improvements
            else {},
        }


# ---------------------------------------------------------------------------
# Sandbox — isolated testing environment
# ---------------------------------------------------------------------------


class Sandbox:
    """Isolated environment for testing improvements before merging.

    Creates a copy of modified files, applies changes, runs tests and evals.
    If tests pass and metrics improve -> ready to merge.
    If not -> revert and mark as failed.
    """

    def __init__(self, sandbox_dir: Path = SANDBOX_DIR):
        self.sandbox_dir = sandbox_dir
        self.backup_dir = sandbox_dir / "_backups"
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.modified_files: list[Path] = []

    def backup_file(self, file_path: Path) -> None:
        """Backup original file before modification."""
        if file_path.exists():
            backup = self.backup_dir / file_path.name
            shutil.copy2(file_path, backup)
            self.modified_files.append(file_path)
            logger.info("Backed up: %s", file_path)

    def apply_code(self, file_path: Path, new_content: str, mode: str = "overwrite") -> None:
        """Apply new code to a file (after backing up original).

        Args:
            mode: "overwrite" replaces file, "append" adds to end (safer).
        """
        self.backup_file(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if mode == "append" and file_path.exists():
            existing = file_path.read_text(encoding="utf-8")
            combined = existing.rstrip() + "\n\n\n" + new_content.strip() + "\n"
            file_path.write_text(combined, encoding="utf-8")
            logger.info("Appended changes to: %s", file_path)
        else:
            file_path.write_text(new_content, encoding="utf-8")
            logger.info("Wrote new file: %s", file_path)

    def run_tests(self) -> dict:
        """Run pytest and return results."""
        logger.info("Running test suite...")
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short", "-q"],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=120,
        )
        passed = result.returncode == 0
        # Parse test counts from output
        output = result.stdout + result.stderr
        test_match = re.search(r"(\d+) passed", output)
        fail_match = re.search(r"(\d+) failed", output)

        return {
            "passed": passed,
            "return_code": result.returncode,
            "tests_passed": int(test_match.group(1)) if test_match else 0,
            "tests_failed": int(fail_match.group(1)) if fail_match else 0,
            "output": output[-2000:],  # Last 2K of output
        }

    def run_eval(self) -> dict:
        """Run eval suite and return metrics."""
        logger.info("Running eval suite...")
        try:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "finetune.eval_suite",
                    "--tasks",
                    "intrinsic,retrieval,latency",
                    "--output",
                    str(self.sandbox_dir / "eval_results.json"),
                ],
                cwd=str(BASE_DIR),
                capture_output=True,
                text=True,
                timeout=120,
            )
            eval_file = self.sandbox_dir / "eval_results.json"
            if eval_file.exists():
                with open(eval_file) as f:
                    return json.load(f)
        except Exception as e:
            logger.warning("Eval failed: %s", e)
        return {}

    def run_syntax_check(self, file_path: Path) -> dict:
        """Quick syntax check on modified file."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                f"import ast; ast.parse(open(r'{file_path}', encoding='utf-8').read())",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return {
            "valid": result.returncode == 0,
            "error": result.stderr.strip() if result.returncode != 0 else "",
        }

    def revert(self) -> None:
        """Revert all modified files to their backups."""
        for file_path in self.modified_files:
            backup = self.backup_dir / file_path.name
            if backup.exists():
                shutil.copy2(backup, file_path)
                logger.info("Reverted: %s", file_path)
            else:
                # New file that didn't exist before -- delete it
                if file_path.exists():
                    file_path.unlink()
                    logger.info("Removed new file: %s", file_path)
        self.modified_files.clear()
        logger.info("All changes reverted")

    def commit_changes(self) -> None:
        """Clean up backups after successful merge."""
        for file_path in self.modified_files:
            backup = self.backup_dir / file_path.name
            if backup.exists():
                backup.unlink()
        self.modified_files.clear()
        logger.info("Changes committed, backups cleaned")


# ---------------------------------------------------------------------------
# Evaluator — decides if improvement is worth keeping
# ---------------------------------------------------------------------------


class ImprovementEvaluator:
    """Evaluates whether an improvement actually helps."""

    @staticmethod
    def compare_metrics(before: dict, after: dict) -> dict:
        """Compare eval metrics before and after an improvement."""
        verdict = {"improved": False, "details": []}

        # Check intrinsic metrics
        if "intrinsic" in before and "intrinsic" in after:
            b_acc = before["intrinsic"].get("ft_accuracy", 0)
            a_acc = after["intrinsic"].get("ft_accuracy", 0)
            b_margin = before["intrinsic"].get("ft_margin", 0)
            a_margin = after["intrinsic"].get("ft_margin", 0)

            if a_acc >= b_acc and a_margin >= b_margin:
                verdict["details"].append(
                    f"Intrinsic: OK (acc {b_acc}->{a_acc}, margin {b_margin}->{a_margin})"
                )
            else:
                verdict["details"].append(
                    f"Intrinsic: DEGRADED (acc {b_acc}->{a_acc}, margin {b_margin}->{a_margin})"
                )
                return verdict

        # Check retrieval metrics
        if "retrieval" in before and "retrieval" in after:
            b_wins = before["retrieval"].get("win_rate", 0)
            a_wins = after["retrieval"].get("win_rate", 0)
            b_mrr = before["retrieval"].get("ft_mrr5", 0)
            a_mrr = after["retrieval"].get("ft_mrr5", 0)

            if a_mrr >= b_mrr:
                verdict["details"].append(
                    f"Retrieval: OK (MRR {b_mrr}->{a_mrr}, wins {b_wins}->{a_wins})"
                )
            else:
                verdict["details"].append(f"Retrieval: DEGRADED (MRR {b_mrr}->{a_mrr})")
                return verdict

        # Check latency
        if "latency" in before and "latency" in after:
            b_lat = before["latency"].get("ft_per_text_ms", 0)
            a_lat = after["latency"].get("ft_per_text_ms", 0)
            overhead = ((a_lat / max(b_lat, 0.01)) - 1) * 100
            if overhead < 50:  # Less than 50% overhead is acceptable
                verdict["details"].append(f"Latency: OK ({b_lat}->{a_lat}ms, {overhead:+.0f}%)")
            else:
                verdict["details"].append(
                    f"Latency: TOO SLOW ({b_lat}->{a_lat}ms, {overhead:+.0f}%)"
                )
                return verdict

        verdict["improved"] = True
        return verdict

    @staticmethod
    def check_obsolete(improvement: dict, current_files: dict[str, str]) -> str | None:
        """Check if an improvement is obsolete (already implemented or superseded).

        Returns reason string if obsolete, None if still relevant.
        """
        hint = (improvement.get("code_snippet") or improvement.get("code_hint", "")).lower()

        # Check if key patterns from the hint are already in target files
        if hint:
            keywords = [
                w
                for w in re.findall(r"\w{5,}", hint)
                if w
                not in (
                    "import",
                    "return",
                    "class",
                    "function",
                    "module",
                    "model",
                    "train",
                    "torch",
                    "numpy",
                    "config",
                    "param",
                )
            ]
            for fname, content in current_files.items():
                content_lower = content.lower()
                matches = sum(1 for kw in keywords if kw in content_lower)
                if matches >= len(keywords) * 0.6:  # 60% of keywords already present
                    return (
                        f"Already implemented in {fname} ({matches}/{len(keywords)} keywords found)"
                    )

        return None


# ---------------------------------------------------------------------------
# Self-Improvement Agent — the full pipeline
# ---------------------------------------------------------------------------


class SelfImproveAgent:
    """The main self-improvement pipeline: analyze -> implement -> test -> merge/revert."""

    INTERESTING_FILES = [
        r".*train.*\.py$",
        r".*lora.*\.py$",
        r".*embed.*\.py$",
        r".*eval.*\.py$",
        r".*optim.*\.py$",
        r".*config.*\.(py|yaml|yml)$",
        r".*loss.*\.py$",
        r".*data.*\.py$",
    ]

    def __init__(self, github_token: str | None = None):
        self.scout = GitHubScout(token=github_token)
        self.extractor = PatternExtractor()
        self.kb = KnowledgeBase()
        self.evaluator = ImprovementEvaluator()

    # --- Phase 1: ANALYZE ---

    def analyze_repo(self, url: str, fetch_files: bool = True) -> RepoAnalysis:
        """Full analysis pipeline for a single repo."""
        logger.info("Analyzing: %s", url)
        info = self.scout.get_repo_info(url)
        logger.info("  %s (%d stars, %s)", info["name"], info["stars"], info["language"])

        readme = self.scout.get_readme(url)
        tree = self.scout.get_tree(url)

        extra_content = ""
        if fetch_files:
            for path in tree:
                if any(re.match(p, path) for p in self.INTERESTING_FILES):
                    content = self.scout.get_file(url, path)
                    if content:
                        extra_content += f"\n--- {path} ---\n{content[:3000]}\n"
                        if len(extra_content) > 10000:
                            break

        improvements_raw = self.extractor.extract(info, readme, tree, extra_content)
        logger.info("  Improvements found: %d", len(improvements_raw))

        analysis = RepoAnalysis(
            url=url,
            name=info.get("name", ""),
            description=info.get("description", ""),
            stars=info.get("stars", 0),
            language=info.get("language", ""),
            readme_summary=readme[:500],
            key_patterns=[i.get("title", "") for i in improvements_raw],
            applicable_improvements=improvements_raw,
            analyzed_at=datetime.now().isoformat(),
        )

        improvements = []
        for i, raw in enumerate(improvements_raw):
            improvements.append(
                Improvement(
                    id=f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{i:02d}",
                    title=raw.get("title", "Untitled"),
                    description=raw.get("description", ""),
                    source_repo=url,
                    category=raw.get("category", "architecture"),
                    priority=raw.get("priority", "medium"),
                    effort=raw.get("effort", "medium"),
                    target_files=raw.get("target_files", []),
                    code_snippet=raw.get("code_hint", ""),
                    status="proposed",
                    created_at=datetime.now().isoformat(),
                )
            )

        self.kb.add_repo(analysis)
        self.kb.add_improvements(improvements)
        self.kb.save()
        return analysis

    def discover(self, query: str, max_repos: int = 5) -> list[RepoAnalysis]:
        """Search GitHub for relevant repos and analyze them."""
        logger.info("Discovering repos for: '%s'", query)
        repos = self.scout.search_repos(query, max_results=max_repos)
        analyses = []
        for repo in repos:
            try:
                analyses.append(self.analyze_repo(repo["url"]))
                time.sleep(2)
            except Exception as e:
                logger.warning("Failed to analyze %s: %s", repo["url"], e)
        return analyses

    # --- Phase 2: PLAN ---

    def get_improvement_plan(self) -> list[dict]:
        """Get prioritized list of proposed improvements."""
        improvements = self.kb.get_improvements(status="proposed")
        priority_order = {"high": 0, "medium": 1, "low": 2}
        effort_order = {"small": 0, "medium": 1, "large": 2}
        improvements.sort(
            key=lambda x: (
                priority_order.get(x["priority"], 1),
                effort_order.get(x["effort"], 1),
            )
        )
        return improvements

    # --- File Mapper ---

    # Maps improvement categories/keywords to OUR actual project files
    PROJECT_FILE_MAP = {
        # Training-related
        "train": "finetune/train.py",
        "trainer": "finetune/train.py",
        "lora": "finetune/lora.py",
        "loss": "finetune/train.py",  # losses are in train.py
        "optim": "finetune/optimizations.py",
        "checkpoint": "finetune/train.py",
        "scheduler": "finetune/train.py",
        "lr": "finetune/train.py",
        "padding": "finetune/optimizations.py",
        "packing": "finetune/optimizations.py",
        "batch": "finetune/optimizations.py",
        "gradient": "finetune/optimizations.py",
        "bf16": "finetune/optimizations.py",
        "fp16": "finetune/optimizations.py",
        "quantiz": "finetune/lora.py",
        # Data-related
        "data": "finetune/data_gen_v2.py",
        "augment": "finetune/data_gen_v2.py",
        "dataset": "finetune/dataset.py",
        "negative": "finetune/data_gen_v2.py",
        # Eval-related
        "eval": "finetune/eval_suite.py",
        "metric": "finetune/eval_suite.py",
        "bench": "bench.py",
        # Architecture
        "config": "finetune/train_config.yaml",
        "embed": "rag/model_registry.py",
        "rerank": "rag/model_registry.py",
        "store": "rag/store.py",
        "index": "rag/index_registry.py",
        "api": "api.py",
        "export": "finetune/export.py",
        "onnx": "finetune/export.py",
        "experiment": "finetune/experiment.py",
    }

    def _resolve_target_file(self, improvement: dict) -> Path | None:
        """Map an improvement's target files to an actual file in our project.

        The LLM suggests target_files from the SOURCE repo (e.g., unsloth/trainer.py).
        We need to map those to OUR project files (e.g., finetune/train.py).
        """
        # 1. Check if any target_files already exist in our project
        for tf in improvement.get("target_files", []):
            fp = BASE_DIR / tf
            if fp.exists():
                return fp

        # 2. Smart mapping by keywords in title, description, and target_files
        search_text = " ".join(
            [
                improvement.get("title", ""),
                improvement.get("description", ""),
                improvement.get("code_snippet", ""),
                " ".join(improvement.get("target_files", [])),
            ]
        ).lower()

        # Score each project file by keyword matches
        scores: dict[str, int] = {}
        for keyword, project_file in self.PROJECT_FILE_MAP.items():
            if keyword in search_text:
                scores[project_file] = scores.get(project_file, 0) + 1

        if scores:
            best_file = max(scores, key=scores.get)
            fp = BASE_DIR / best_file
            if fp.exists():
                logger.info("Mapped target to: %s (score: %d)", best_file, scores[best_file])
                return fp
            # File doesn't exist yet — create it (e.g., finetune/export.py)
            logger.info("Will create new file: %s", best_file)
            return fp

        # 3. Fallback: finetune/train.py for training, optimizations.py for optimization
        category = improvement.get("category", "")
        fallback = {
            "training": "finetune/train.py",
            "optimization": "finetune/optimizations.py",
            "data": "finetune/data_gen_v2.py",
            "eval": "finetune/eval_suite.py",
            "architecture": "rag/config.py",
        }.get(category, "finetune/train.py")

        logger.info("Fallback target: %s (category: %s)", fallback, category)
        return BASE_DIR / fallback

    # --- Phase 3: IMPLEMENT + TEST + EVALUATE ---

    def implement_improvement(self, imp_id: str) -> dict:
        """Implement a single improvement: generate code -> test -> eval -> decide.

        Returns result dict with status and details.
        """
        # Find improvement
        improvement = None
        for imp in self.kb.data["improvements"]:
            if imp["id"] == imp_id:
                improvement = imp
                break

        if not improvement:
            return {"status": "error", "message": f"Improvement {imp_id} not found"}

        logger.info("=" * 60)
        logger.info("IMPLEMENTING: %s", improvement["title"])
        logger.info("=" * 60)

        sandbox = Sandbox()

        # Step 0: Check if already obsolete
        current_files = {}
        for tf in improvement.get("target_files", []):
            fp = BASE_DIR / tf
            if fp.exists():
                current_files[tf] = fp.read_text(encoding="utf-8")

        obsolete_reason = self.evaluator.check_obsolete(improvement, current_files)
        if obsolete_reason:
            logger.info("OBSOLETE: %s", obsolete_reason)
            self.kb.update_improvement(imp_id, status="obsolete", rejection_reason=obsolete_reason)
            return {"status": "obsolete", "reason": obsolete_reason}

        self.kb.update_improvement(imp_id, status="implementing")

        # Step 1: Get baseline eval
        logger.info("Step 1: Getting baseline eval...")
        eval_before = sandbox.run_eval()

        # Step 2: Resolve target file (map from source repo to our project)
        actual_target = self._resolve_target_file(improvement)
        if not actual_target:
            self.kb.update_improvement(
                imp_id, status="failed", rejection_reason="Could not resolve target file"
            )
            return {"status": "failed", "reason": "No target file"}

        logger.info("Step 2: Generating code for %s...", actual_target)
        context_files = [
            BASE_DIR / "finetune" / "train.py",
            BASE_DIR / "finetune" / "lora.py",
            BASE_DIR / "finetune" / "optimizations.py",
        ]

        new_code = self.extractor.generate_implementation(improvement, actual_target, context_files)

        if not new_code:
            self.kb.update_improvement(
                imp_id, status="failed", rejection_reason="LLM failed to generate code"
            )
            return {"status": "failed", "reason": "Code generation failed"}

        # Clean code (remove markdown fences if any)
        new_code = new_code.strip()
        if new_code.startswith("```"):
            new_code = re.sub(r"^```\w*\n?", "", new_code)
            new_code = re.sub(r"\n?```$", "", new_code)

        # Step 3: Syntax check
        logger.info("Step 3: Syntax check...")
        sandbox.apply_code(actual_target, new_code, mode="append")
        syntax = sandbox.run_syntax_check(actual_target)

        if not syntax["valid"]:
            logger.warning("SYNTAX ERROR: %s", syntax["error"])
            sandbox.revert()
            self.kb.update_improvement(
                imp_id, status="failed", rejection_reason=f"Syntax error: {syntax['error'][:200]}"
            )
            return {"status": "failed", "reason": "Syntax error", "error": syntax["error"]}

        # Step 4: Run tests
        logger.info("Step 4: Running tests...")
        self.kb.update_improvement(imp_id, status="testing")
        test_results = sandbox.run_tests()

        if not test_results["passed"]:
            logger.warning(
                "TESTS FAILED: %d passed, %d failed",
                test_results["tests_passed"],
                test_results["tests_failed"],
            )
            sandbox.revert()
            self.kb.update_improvement(
                imp_id,
                status="failed",
                test_results=test_results,
                rejection_reason=f"Tests failed: {test_results['tests_failed']} failures",
            )
            return {"status": "failed", "reason": "Tests failed", "test_results": test_results}

        logger.info(
            "Tests passed: %d/%d",
            test_results["tests_passed"],
            test_results["tests_passed"] + test_results["tests_failed"],
        )

        # Step 5: Run eval
        logger.info("Step 5: Running eval suite...")
        eval_after = sandbox.run_eval()

        # Step 6: Compare metrics
        logger.info("Step 6: Comparing metrics...")
        comparison = self.evaluator.compare_metrics(eval_before, eval_after)

        for detail in comparison["details"]:
            logger.info("  %s", detail)

        if comparison["improved"]:
            # SUCCESS - keep changes
            logger.info("IMPROVEMENT ACCEPTED - metrics improved!")
            sandbox.commit_changes()
            self.kb.update_improvement(
                imp_id,
                status="merged",
                implemented_at=datetime.now().isoformat(),
                test_results=test_results,
                eval_before=eval_before,
                eval_after=eval_after,
            )
            return {
                "status": "merged",
                "test_results": test_results,
                "eval_comparison": comparison,
            }
        else:
            # FAILED - revert
            logger.info("IMPROVEMENT REJECTED - metrics did not improve")
            sandbox.revert()
            self.kb.update_improvement(
                imp_id,
                status="rejected",
                test_results=test_results,
                eval_before=eval_before,
                eval_after=eval_after,
                rejection_reason="Metrics did not improve: " + "; ".join(comparison["details"]),
            )
            return {
                "status": "rejected",
                "reason": "Metrics did not improve",
                "eval_comparison": comparison,
            }

    # --- Phase 4: AUTO mode ---

    def auto_improve(self, max_improvements: int = 3) -> list[dict]:
        """Full autonomous loop: pick top improvements, implement, test, decide.

        Returns list of results for each attempted improvement.
        """
        plan = self.get_improvement_plan()
        if not plan:
            logger.info("No proposed improvements to implement")
            return []

        results = []
        attempted = 0

        for imp in plan:
            if attempted >= max_improvements:
                break

            # Skip large-effort improvements in auto mode
            if imp["effort"] == "large":
                logger.info("Skipping large-effort: %s", imp["title"])
                continue

            logger.info("\n" + "=" * 60)
            logger.info("AUTO-IMPROVE [%d/%d]: %s", attempted + 1, max_improvements, imp["title"])
            logger.info("=" * 60)

            result = self.implement_improvement(imp["id"])
            result["improvement_title"] = imp["title"]
            results.append(result)
            attempted += 1

            # Brief pause between improvements
            time.sleep(1)

        # Summary
        merged = [r for r in results if r["status"] == "merged"]
        failed = [r for r in results if r["status"] == "failed"]
        rejected = [r for r in results if r["status"] == "rejected"]
        obsolete = [r for r in results if r["status"] == "obsolete"]

        logger.info("\n" + "=" * 60)
        logger.info("AUTO-IMPROVE SUMMARY")
        logger.info("  Attempted: %d", len(results))
        logger.info("  Merged: %d", len(merged))
        logger.info("  Rejected: %d", len(rejected))
        logger.info("  Failed: %d", len(failed))
        logger.info("  Obsolete: %d", len(obsolete))
        logger.info("=" * 60)

        return results

    # --- Display ---

    def print_plan(self) -> None:
        plan = self.get_improvement_plan()
        stats = self.kb.get_stats()

        print("\n" + "=" * 70)
        print("SELF-IMPROVEMENT PLAN")
        print(
            f"Repos analyzed: {stats['total_repos']} | Improvements: {stats['total_improvements']}"
        )
        if stats.get("by_status"):
            parts = [f"{s}: {n}" for s, n in stats["by_status"].items()]
            print(f"Status: {' | '.join(parts)}")
        print("=" * 70)

        for i, imp in enumerate(plan, 1):
            tag = {"high": "[!!!]", "medium": "[!!]", "low": "[!]"}.get(imp["priority"], "[?]")
            effort = {"small": "quick", "medium": "moderate", "large": "heavy"}.get(
                imp["effort"], ""
            )
            print(f"\n{i}. {tag} [{imp['category'].upper()}] {imp['title']}")
            print(f"   {imp['description'][:120]}")
            print(f"   Effort: {effort} | Source: {imp['source_repo'].split('/')[-1]}")
            print(f"   ID: {imp['id']}")
            if imp.get("target_files"):
                print(f"   Files: {', '.join(imp['target_files'][:3])}")

        print("\n" + "=" * 70)

    def print_history(self) -> None:
        """Show all improvements with their final status."""
        all_imps = self.kb.data["improvements"]

        print("\n" + "=" * 70)
        print("IMPROVEMENT HISTORY")
        print("=" * 70)

        for imp in all_imps:
            status_tag = {
                "proposed": "[PENDING]",
                "merged": "[MERGED]",
                "rejected": "[REJECTED]",
                "failed": "[FAILED]",
                "obsolete": "[OBSOLETE]",
            }.get(imp["status"], f"[{imp['status'].upper()}]")

            print(f"\n{status_tag} {imp['title']}")
            if imp.get("rejection_reason"):
                print(f"   Reason: {imp['rejection_reason'][:100]}")
            if imp.get("implemented_at"):
                print(f"   Merged at: {imp['implemented_at']}")

        print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Self-improvement agent")
    parser.add_argument("--repo", type=str, help="Analyze a specific GitHub repo URL")
    parser.add_argument("--discover", type=str, help="Search and analyze repos by query")
    parser.add_argument("--max-repos", type=int, default=5)
    parser.add_argument("--plan", action="store_true", help="Show current improvement plan")
    parser.add_argument("--history", action="store_true", help="Show all improvement history")
    parser.add_argument("--stats", action="store_true", help="Show knowledge base stats")
    parser.add_argument("--implement", type=str, help="Implement a specific improvement by ID")
    parser.add_argument(
        "--auto", action="store_true", help="Auto-improve: implement top improvements"
    )
    parser.add_argument("--max-auto", type=int, default=3, help="Max improvements in auto mode")
    parser.add_argument("--token", type=str, default=None, help="GitHub token")
    args = parser.parse_args()

    agent = SelfImproveAgent(github_token=args.token)

    if args.stats:
        print(json.dumps(agent.kb.get_stats(), indent=2))
        return

    if args.plan:
        agent.print_plan()
        return

    if args.history:
        agent.print_history()
        return

    if args.repo:
        analysis = agent.analyze_repo(args.repo)
        print(f"\nAnalyzed: {analysis.name} ({analysis.stars} stars)")
        print(f"Patterns found: {len(analysis.key_patterns)}")
        for p in analysis.key_patterns:
            print(f"  - {p}")
        agent.print_plan()
        return

    if args.discover:
        analyses = agent.discover(args.discover, max_repos=args.max_repos)
        print(f"\nDiscovered and analyzed {len(analyses)} repos")
        agent.print_plan()
        return

    if args.implement:
        result = agent.implement_improvement(args.implement)
        print(f"\nResult: {json.dumps(result, indent=2, default=str)}")
        return

    if args.auto:
        results = agent.auto_improve(max_improvements=args.max_auto)
        print(f"\nAuto-improve complete: {len(results)} attempted")
        for r in results:
            print(f"  [{r['status'].upper()}] {r.get('improvement_title', 'unknown')}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
