# herramientas_fraude_avanzado_v2.py
"""
Herramientas de fraude - versión universal
- Intenta usar MCP FastMCP si está disponible.
- Si no, expone un HTTP JSON endpoint /call (POST) en localhost:8000
  para invocar las mismas funciones.
- También permite invocación por CLI.

Uso (ejemplos):
1) Ejecutar servidor (MCP o fallback HTTP):
   python herramientas_fraude_avanzado.py --serve

2) Entrenar modelo sintético (CLI):
   python herramientas_fraude_avanzado.py --entrenar_sintetico

3) Predecir un reclamo (CLI):
   python herramientas_fraude_avanzado.py --predecir "Paciente con dolor crónico..."

4) Llamada HTTP (fallback server):
   curl -X POST http://localhost:8000/call -H "Content-Type: application/json" \
     -d '{"tool":"detectar_fraude_heuristico","args":{"reclamo":"Paciente..."} }'
"""

import json
import os
import re
import sqlite3
import argparse
import traceback
from typing import Any, Dict

# ---------- Optional heavy deps guarded ----------
try:
    import numpy as np
except Exception:
    np = None

try:
    import joblib
except Exception:
    joblib = None

# sklearn optional
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# sentence-transformers optional
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except Exception:
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# ---------- Try to import FastMCP (new MCP API) ----------
USE_FASTMCP = False
try:
    # may raise ImportError
    from mcp.server.fastmcp import FastMCP
    USE_FASTMCP = True
except Exception:
    USE_FASTMCP = False

# ---------- Config ----------
DB_PATH = "ejemplo_reclamos.db"
MODELO_FRAUDE_PATH = "modelo_fraude.pkl"
VECTORIZER_PATH = "vectorizer.pkl"
EMBED_MODEL_NAME = "LaBSE"   #"paraphrase-multilingual-MiniLM-L12-v2"
# "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# ---------- Heuristics data ----------
PALABRAS_SOSPECHOSAS = [
    "accidente", "dolor crónico", "dolor","incapacidad total",
    "pérdida auditiva", "sordera", "hombro congelado",
    "lumbago severo", "incapacidad permanente", "caída",
    "discapacidad","severo","pronóstico reservado","secuelas",
    "rehabilitación prolongada","latigazo cervical",
    "síndrome del túnel carpiano","fibromialgia","estrés postraumático",
    "recurrente","múltiples citas","sesiones diarias","traumatismo"
]

DIAGNOSTICOS_CRITICOS_PREFIJOS = ["H90", "K65", "S33","G89", "F43", "T07", "M54"]
PROCEDIMIENTOS_CAROS = ["47562", "22857", "33405","33249", "63030", "77067", "99205", "99215", "88305"]


# ---------------- Utilities ----------------
def normalizar_texto(t: str) -> str:
    if t is None:
        return ""
    t = str(t).lower()
    t = re.sub(r"[^a-z0-9áéíóúüñ\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS reclamos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            reclamo TEXT,
            prob_heuristica REAL,
            prob_modelo REAL,
            created TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


# ---------------- Core functions ----------------
def extraer_codigos_cie(texto: str):
    texto_up = (texto or "").upper()
    simples = re.findall(r"\b[A-Z]\d{1,3}\b", texto_up)
    puntos = re.findall(r"\b[A-Z]\d{1,3}\.\d+\b", texto_up)
    return list(dict.fromkeys(simples + puntos))


def extraer_cpt(texto: str):
    return re.findall(r"\b\d{4,6}\b", texto or "")


def detectar_fraude_heuristico(reclamo: str) -> Dict[str, Any]:
    """
    Heurística simple que devuelve una probabilidad (0-1) y señales.
    """
    texto = normalizar_texto(reclamo)
    kw = [w for w in PALABRAS_SOSPECHOSAS if w in texto]
    score_kw = min(len(kw) * 0.25, 0.75)

    codigos = extraer_codigos_cie(reclamo)
    score_dx = 0.0
    for c in codigos:
        for pref in DIAGNOSTICOS_CRITICOS_PREFIJOS:
            if c.startswith(pref):
                score_dx += 0.4
    score_dx = min(score_dx, 1.0)

    cpt = extraer_cpt(reclamo)
    score_cpt = 0.4 if any(len(x) >= 5 for x in cpt) else 0.0

    # similaridad semantica usando transformers
    score_sem = 0.0
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        try:
            model = SentenceTransformer(EMBED_MODEL_NAME)
            emb_text = model.encode([texto], show_progress_bar=False)[0]
            seeds = ["fraude", "reclamo sospechoso", "cobro indebido", "incapacidad permanente",
                    "diagnóstico falso", "procedimiento no realizado","permanente", "dolor crónico severo",
                    "extremos", "invalidez total", "engaño", "simulación", "declaración falsa",
                    "manipulación de documentos", "sobre-facturación", "costos exorbitantes",
                    "tratamiento excesivo", "discapacidad", "falsificado", "inventado", "exagerado"]
            emb_seed = model.encode(seeds, show_progress_bar=False).mean(axis=0)
            sim = float(np.dot(emb_text, emb_seed) / (np.linalg.norm(emb_text) * np.linalg.norm(emb_seed)))
            score_sem = max(0.0, (sim - 0.2))
            score_sem = min(score_sem, 1.0)
        except Exception:
            score_sem = 0.0

    prob = (0.35 * score_kw) + (0.35 * score_dx) + (0.2 * score_cpt) + (0.1 * score_sem)
    prob = float(min(max(prob, 0.0), 1.0))

    return {
        "probabilidad_heuristica": round(prob, 3),
        "keywords_detectadas": kw,
        "codigos_cie": codigos,
        "procedimientos_detectados": cpt,
        "score_kw": round(score_kw, 3),
        "score_dx": round(score_dx, 3),
        "score_cpt": round(score_cpt, 3),
        "score_sem": round(score_sem, 3),
    }


def entrenar_modelo_sintetico(n_samples: int = 500, random_state: int = 42) -> Dict[str, Any]:
    """
    Genera datos sintéticos y entrena un logistic regression.
    Requiere scikit-learn. Guarda modelo y vectorizer en disco.
    """
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn no está disponible. Instala scikit-learn para entrenar.")

    rng = np.random.RandomState(random_state)
    montos = rng.exponential(scale=2000, size=n_samples)
    edades = rng.randint(18, 90, size=n_samples)
    dias = rng.randint(0, 365, size=n_samples)
    cnt_kw = rng.poisson(0.3, size=n_samples)
    has_cpt5 = rng.binomial(1, 0.08, size=n_samples)
    y = ((montos > 4000) & (dias > 60)) | ((has_cpt5 == 1) & (cnt_kw > 0))
    y = y.astype(int)
    X = np.vstack([montos, edades, dias, cnt_kw, has_cpt5]).T
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    prob = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, prob) if len(set(y_test)) > 1 else 0.5
    if joblib:
        joblib.dump(clf, MODELO_FRAUDE_PATH)
    return {"n_samples": n_samples, "auc_test": float(round(auc, 4)), "modelo_guardado": MODELO_FRAUDE_PATH}


def predecir_modelo(monto: float, edad: int, dias_incapacidad: int, cnt_keywords: int = 0, has_cpt5: int = 0) -> Dict[str, Any]:
    """
    Predice usando el modelo entrenado sintético.
    """
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn no está disponible. Instala scikit-learn para predecir.")
    if not joblib or not os.path.exists(MODELO_FRAUDE_PATH):
        return {"error": f"{MODELO_FRAUDE_PATH} no existe. Ejecuta entrenar_modelo_sintetico primero."}
    modelo = joblib.load(MODELO_FRAUDE_PATH)
    X = np.array([[monto, edad, dias_incapacidad, cnt_keywords, has_cpt5]], dtype=float)
    prob = float(modelo.predict_proba(X)[0][1])
    return {"probabilidad_modelo": round(prob, 3)}


def guardar_reclamo(reclamo: str, prob_heuristica: float = None, prob_modelo: float = None) -> Dict[str, Any]:
    _init_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO reclamos (reclamo, prob_heuristica, prob_modelo) VALUES (?, ?, ?)",
        (reclamo, prob_heuristica, prob_modelo),
    )
    conn.commit()
    conn.close()
    return {"status": "ok"}


def consultar_reclamos(limit: int = 20):
    _init_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, reclamo, prob_heuristica, prob_modelo, created FROM reclamos ORDER BY id DESC LIMIT ?", (int(limit),))
    rows = cur.fetchall()
    conn.close()
    cols = ["id", "reclamo", "prob_heuristica", "prob_modelo", "created"]
    return [dict(zip(cols, r)) for r in rows]


# ---------------- Mostrar el listado de funciones ----------------
TOOLS = {
    "detectar_fraude_heuristico": detectar_fraude_heuristico,
    "entrenar_modelo_sintetico": entrenar_modelo_sintetico,
    "predecir_modelo": predecir_modelo,
    "guardar_reclamo": guardar_reclamo,
    "consultar_reclamos": consultar_reclamos,
}

# ---------------- MCP si estuviese disponible ----------------
if USE_FASTMCP:
    try:
        mcp = FastMCP("fraude_avanzado")
        # register tools automatically
        for name, fn in TOOLS.items():
            # decorator-like attach
            mcp.tool(name=name)(fn)  # some FastMCP versions accept this form
        # run only if executed as main and --serve provided; see below
        _mcp_ready = True
    except Exception:
        USE_FASTMCP = False
        _mcp_ready = False
else:
    _mcp_ready = False

# ---------------- Fallback HTTP server (no external deps) ----------------
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading

class SimpleJSONHandler(BaseHTTPRequestHandler):
    def _set_headers(self, code=200):
        self.send_response(code)
        self.send_header("Content-type", "application/json")
        self.end_headers()

    def do_GET(self):
        if self.path == "/":
            self._set_headers()
            self.wfile.write(json.dumps({"ok": True, "tools": list(TOOLS.keys())}).encode())
        else:
            self.send_error(404, "Not found")

    def do_POST(self):
        if self.path != "/call":
            self.send_error(404, "Only /call available")
            return
        try:
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length).decode() if length else ""
            data = json.loads(body) if body else {}
            tool = data.get("tool")
            args = data.get("args", {})

            if tool not in TOOLS:
                self._set_headers(400)
                self.wfile.write(json.dumps({"error": f"tool '{tool}' not found", "available": list(TOOLS.keys())}).encode())
                return

            fn = TOOLS[tool]
            # call function (supports both positional in list or kwargs dict)
            if isinstance(args, list):
                result = fn(*args)
            elif isinstance(args, dict):
                result = fn(**args)
            else:
                result = fn(args)

            # make sure result is JSON serializable
            self._set_headers(200)
            self.wfile.write(json.dumps({"ok": True, "result": result}, default=str).encode())
        except Exception as e:
            self._set_headers(500)
            tb = traceback.format_exc()
            self.wfile.write(json.dumps({"error": str(e), "traceback": tb}).encode())

def start_fallback_http_server(host="127.0.0.1", port=8000):
    print(f"Starting fallback HTTP server at http://{host}:{port} (tools: {list(TOOLS.keys())})")
    server = HTTPServer((host, port), SimpleJSONHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("HTTP server stopped.")

# ---------------- CLI helpers ----------------
def call_tool_cli(tool_name: str, args: Dict[str, Any]):
    if tool_name not in TOOLS:
        raise ValueError(f"Tool {tool_name} not found. Available: {list(TOOLS.keys())}")
    fn = TOOLS[tool_name]
    if isinstance(args, dict):
        return fn(**args)
    elif isinstance(args, list):
        return fn(*args)
    else:
        return fn(args)

# ---------------- Main entrypoint ----------------
def main():
    parser = argparse.ArgumentParser(description="Herramientas fraude - servidor/CLI")
    parser.add_argument("--serve", action="store_true", help="Levantar servidor MCP (si disponible) o fallback HTTP")
    parser.add_argument("--entrenar_sintetico", action="store_true", help="Entrena modelo sintético")
    parser.add_argument("--predecir", type=str, help="Texto para predecir (usa heurística + modelo si existe)")
    parser.add_argument("--tool", type=str, help="Invocar tool por nombre (CLI)")
    parser.add_argument("--args", type=str, help="JSON string con args para tool")
    parser.add_argument("--port", type=int, default=8000, help="Puerto para fallback HTTP server")
    args = parser.parse_args()

    if args.serve:
        if _mcp_ready:
            print("FastMCP detected — iniciando MCP server (FastMCP)...")
            # run the MCP server; FastMCP.run may block
            try:
                mcp.run()
            except Exception as e:
                print("Error arrancando FastMCP:", e)
                print("Cayendo al fallback HTTP server.")
                start_fallback_http_server(port=args.port)
        else:
            start_fallback_http_server(port=args.port)
        return

    if args.entrenar_sintetico:
        print("Entrenando modelo sintético...")
        try:
            res = entrenar_modelo_sintetico()
            print("Resultado:", res)
        except Exception as e:
            print("Error entrenando:", e)
            traceback.print_exc()
        return

    if args.predecir:
        texto = args.predecir
        print("Heurística:")
        h = detectar_fraude_heuristico(texto)
        print(json.dumps(h, indent=2, ensure_ascii=False))
        # try model prediction if model exists
        if SKLEARN_AVAILABLE and joblib and os.path.exists(MODELO_FRAUDE_PATH):
            print("Modelo supervisado (si existe):")
            try:
                # best-effort: try to estimate cnt_keywords and has_cpt5
                cnt_kw = sum(1 for w in PALABRAS_SOSPECHOSAS if w in normalizar_texto(texto))
                has_cpt5 = int(any(len(x) >= 5 for x in extraer_cpt(texto)))
                # default numbers for monto/edad/dias - user can call predecir_modelo tool for full control
                sample = predecir_modelo(monto=5000, edad=45, dias_incapacidad=90, cnt_keywords=cnt_kw, has_cpt5=has_cpt5)
                print(json.dumps(sample, indent=2, ensure_ascii=False))
            except Exception:
                pass
        return

    if args.tool:
        try:
            parsed_args = {}
            if args.args:
                parsed_args = json.loads(args.args)
            res = call_tool_cli(args.tool, parsed_args)
            print(json.dumps({"ok": True, "result": res}, indent=2, ensure_ascii=False, default=str))
        except Exception as e:
            print("Error calling tool:", e)
            traceback.print_exc()
        return

    # default help
    print(__doc__)


if __name__ == "__main__":
    main()
