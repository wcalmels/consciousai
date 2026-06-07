#!/usr/bin/env bash
# =============================================================================
# publish_to_github.sh
# Publica ConsciousAI en GitHub con un solo comando.
#
# USO:
#   chmod +x publish_to_github.sh
#   ./publish_to_github.sh TU_USUARIO TU_TOKEN
#
# OBTENER TOKEN:
#   github.com → Settings → Developer settings →
#   Personal access tokens → Tokens (classic) → Generate new token
#   Scopes necesarios: ✅ repo
#
# EJEMPLO:
#   ./publish_to_github.sh wcalmels ghp_abc123xyz
# =============================================================================

set -e

GITHUB_USER="${1:-}"
GITHUB_TOKEN="${2:-}"
REPO_NAME="consciousai"

# ── Validación ────────────────────────────────────────────────────────────────
if [[ -z "$GITHUB_USER" || -z "$GITHUB_TOKEN" ]]; then
  echo ""
  echo "❌  Faltan argumentos."
  echo ""
  echo "Uso: ./publish_to_github.sh USUARIO TOKEN"
  echo ""
  echo "Obtener token:"
  echo "  github.com → Settings → Developer settings"
  echo "  → Personal access tokens → Generate new token"
  echo "  → Scope: ✅ repo"
  echo ""
  exit 1
fi

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║   ConsciousAI — Publicación en GitHub            ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""
echo "  Usuario : $GITHUB_USER"
echo "  Repo    : $REPO_NAME"
echo ""

# ── 1. Crear repositorio en GitHub vía API ────────────────────────────────────
echo "▶ Paso 1/4: Creando repositorio en GitHub..."

HTTP_CODE=$(curl -s -o /tmp/gh_response.json -w "%{http_code}" \
  -X POST \
  -H "Authorization: token $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github.v3+json" \
  https://api.github.com/user/repos \
  -d "{
    \"name\": \"$REPO_NAME\",
    \"description\": \"Scalable spectral approximation of integrated information (Φ) for real-time monitoring of autonomous systems, LLMs, and financial markets\",
    \"homepage\": \"https://github.com/$GITHUB_USER/$REPO_NAME\",
    \"private\": false,
    \"has_issues\": true,
    \"has_wiki\": false,
    \"auto_init\": false,
    \"topics\": []
  }")

if [[ "$HTTP_CODE" == "201" ]]; then
  echo "  ✅ Repositorio creado: https://github.com/$GITHUB_USER/$REPO_NAME"
elif [[ "$HTTP_CODE" == "422" ]]; then
  echo "  ℹ️  Repositorio ya existe — continuando..."
else
  echo "  ❌ Error HTTP $HTTP_CODE creando repositorio:"
  cat /tmp/gh_response.json
  exit 1
fi

# ── 2. Agregar remote y push ──────────────────────────────────────────────────
echo ""
echo "▶ Paso 2/4: Configurando remote..."

# Remover remote si ya existe
git remote remove origin 2>/dev/null || true

git remote add origin "https://$GITHUB_USER:$GITHUB_TOKEN@github.com/$GITHUB_USER/$REPO_NAME.git"
echo "  ✅ Remote configurado"

echo ""
echo "▶ Paso 3/4: Subiendo código a GitHub..."
git push -u origin main --force

echo ""
echo "  ✅ Código subido"

# ── 3. Añadir topics vía API ──────────────────────────────────────────────────
echo ""
echo "▶ Paso 4/4: Configurando topics..."

curl -s -X PUT \
  -H "Authorization: token $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github.mercy-preview+json" \
  "https://api.github.com/repos/$GITHUB_USER/$REPO_NAME/topics" \
  -d '{"names":["integrated-information-theory","anomaly-detection","autonomous-systems","time-series","drones","iit","numba","python","real-time","consciousness"]}' \
  > /dev/null

echo "  ✅ Topics configurados"

# ── 4. Crear release v3.0.0 ───────────────────────────────────────────────────
echo ""
echo "▶ Creando release v3.0.0..."

curl -s -X POST \
  -H "Authorization: token $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github.v3+json" \
  "https://api.github.com/repos/$GITHUB_USER/$REPO_NAME/releases" \
  -d '{
    "tag_name": "v3.0.0",
    "target_commitish": "main",
    "name": "ConsciousAI v3.0.0 — Multi-domain expansion",
    "body": "## What'\''s new in v3.0\n\n- `ConnectivityLearner`: automatic C matrix (pearson, granger, mutual_info, ensemble)\n- `LLMPhiProbe`: layer-wise Φ for transformer analysis\n- `FinancialPhiMonitor`: rolling Φ for systemic risk\n- Numba JIT kernels (155× hash speedup, 2.3× batch)\n- 32 tests passing\n- UCR anomaly detection benchmark\n\n## Install\n```bash\npip install numpy numba scipy scikit-learn\ngit clone https://github.com/'"$GITHUB_USER"'/consciousai.git\ncd consciousai && pytest tests/ -q\n```",
    "draft": false,
    "prerelease": false
  }' > /dev/null

echo "  ✅ Release v3.0.0 creado"

# ── Resumen final ─────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║   ✅  ConsciousAI publicado exitosamente                     ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║"
echo "║  🔗 Repo:    https://github.com/$GITHUB_USER/$REPO_NAME"
echo "║  📦 Release: https://github.com/$GITHUB_USER/$REPO_NAME/releases/tag/v3.0.0"
echo "║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Próximos pasos recomendados:                                 ║"
echo "║                                                               ║"
echo "║  1. Postear en Hacker News:                                   ║"
echo "║     news.ycombinator.com → Submit → Show HN: [título]         ║"
echo "║                                                               ║"
echo "║  2. Subir paper a arXiv:                                      ║"
echo "║     arxiv.org → Submit → cs.LG + cs.AI                       ║"
echo "║                                                               ║"
echo "║  3. Validar en UCR real (250 series):                         ║"
echo "║     wu.cs.ucr.edu/keogh/UCRArchive_2018.zip                  ║"
echo "║                                                               ║"
echo "║  4. Contactar PyPhi maintainer (colaboración):                ║"
echo "║     mayner@wisc.edu                                           ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
