# colibrì 전수조사 분석 및 활용·수익화 정리 (한국어)

> 작성일: 2026-10-08
> 분석 대상 커밋: `2c2855e` / 버전 **v1.11.0** (2026-09-13)
> 분석 방식: 저장소 전체 폴더·파일 전수조사 (코드 약 20만 줄, 문서 45종)

## 📎 GitHub 주소

| 구분 | 주소 |
|---|---|
| **원본(업스트림) 저장소** | https://github.com/JustVugg/colibri |
| **현재 분석/작업 저장소 (포크)** | https://github.com/bmshin94/colibri |
| 공식 웹사이트 | https://justvugg.github.io/colibri |
| 릴리스(프리빌드 바이너리) | https://github.com/JustVugg/colibri/releases |
| 이슈 트래커 | https://github.com/JustVugg/colibri/issues |
| Discord 커뮤니티 | https://discord.gg/RXV83nSZdk |
| GLM-5.2 int4 모델(권장 컨테이너) | https://huggingface.co/mastouri/GLM-5.2-colibri-int4-g64-with-int8-mtp |

업스트림 현황(조사 시점): ⭐ **40,376 stars** / 🍴 **4,440 forks** / 생성 2026-07-01 / 주 언어 **C** / 라이선스 **Apache-2.0**

---

## 1. 이게 뭐하는 프로젝트인가

### 한 줄 정의

**colibrì(콜리브리)** = 744B~2.8T 파라미터급 초거대 **MoE(전문가 혼합)** 모델을, GPU 없는 평범한 개인 PC에서 **디스크(NVMe SSD)에서 전문가를 실시간 스트리밍**해 구동하는 **순수 C 추론 엔진**.

슬로건: *"Tiny engine, immense model"* (작은 엔진, 거대한 모델)

### 핵심 아이디어

744B MoE 모델은 **토큰 1개당 약 40B(5.4%)만 활성화**되고, 그중 매 토큰 변하는 부분은 약 11 GB뿐입니다.

| 항목 | 수치 |
|---|---|
| 전체 파라미터 | 744B |
| 토큰당 활성화 | ~40B (**5.4%**) |
| 상시 상주 dense 부분 (어텐션·공유전문가·임베딩) | ~17B → int4로 **9.9 GB** |
| 라우팅 전문가 개수 | **19,456개** (75 MoE 레이어 × 256 + MTP 헤드) |
| 전문가 1개 크기 | 약 19 MB (int4) |
| 전문가 전체 디스크 | 약 372 GB |

그래서 모델을 "RAM에 전부 넣는 것"이 아니라 **"계층에 배치(placement)"** 합니다.

```
[ VRAM ]  가장 빠름 — 가장 뜨거운 전문가 (GPU 있을 때)
[  RAM ]  dense 9.9 GB 상주 + 전문가 LRU 캐시 + 학습된 pin set
[ NVMe ]  19,456개 전문가 전체 (~372 GB) — 필요할 때 스트리밍
```

프로젝트는 이를 **"가중치를 위한 JIT 컴파일러"** 로 설명합니다. JIT가 뜨거운 경로만 컴파일하듯, colibrì는 라우터가 필요성을 증명한 전문가만 그 순간에 끌어옵니다. 쓸수록 내 워크로드에 맞는 전문가가 뜨거워져 점점 빨라집니다(학습된 핫스토어 + 한 레이어 앞선 prefetch).

### 공식 설계 원칙

- **속도 SLA는 없고, 의미(semantics) 보장은 확실하다.** 빠른 메모리가 부족하면 **느려질 수는 있어도, 모델의 정밀도나 라우터 의미를 조용히 바꾸지 않는다.**
- 최적화는 **재현 가능한 end-to-end 측정으로 자격을 증명**해야 한다. 네거티브 결과도 가치가 있다.

---

## 2. 폴더 전수조사

```
colibri/
├── c/                        ★ 엔진 본체
│   ├── colibri.c    12,237줄  GLM-5.2 엔진 (make glm)
│   ├── deepseek_v4.c 18,348줄 DeepSeek V4 Flash
│   ├── kimi_k3.c      4,084줄 Kimi K3 (2.8T)
│   ├── qwen36.c / qwen38.c / glm53.c / inkling.c / olmoe.c / deepseek_v41.c
│   │                         → "모델 1종 = C 파일 1개" 원칙 (9개 패밀리)
│   ├── st.h                  safetensors 인덱스 + range read
│   ├── quant.h      2,102줄  int4/fp4/fp8/mxfp4 컨테이너 디코더
│   ├── tok.h                 tiktoken BPE 정확 재구현 (merges 파일 불필요)
│   ├── expert_store.h        ★ 전문가 스트리밍 캐시
│   ├── route_trace.h         라우팅 텔레메트리(.coli_usage), 엔진 공통
│   ├── kv_prefix.h / kv_persist.h / kv_fp8.h / kv_tq.h   KV 재사용·압축·영속
│   ├── grammar.h / schema_gbnf.h   GBNF 문법 강제 출력(구조화 출력)
│   ├── uring.h / fused_simd.h / omp_tune.h  io_uring, SIMD, OpenMP 튜닝
│   ├── backend_cuda.cu    2,950줄  NVIDIA (+ backend_cuda_dsv4.cu 2,369줄)
│   ├── backend_metal.mm   2,039줄  Apple Silicon
│   ├── backend_vulkan.c   2,166줄  AMD/Intel/모든 GPU (ROCm 중단 카드 포함)
│   ├── coli                  ★ 사용자 CLI (Python 런처)
│   ├── openai_server.py  4,620줄  OpenAI + Anthropic 호환 HTTP 게이트웨이
│   ├── resource_plan.py  1,258줄  RAM/VRAM 플래너 (coli plan / doctor)
│   ├── doctor.py           735줄  진단
│   ├── autotune.py         617줄  최적 실행 프로파일 자동 측정
│   ├── family_registry.py 1,541줄 모델 패밀리 레지스트리
│   ├── Makefile          1,976줄  플랫폼별 빌드 전부
│   ├── tools/                모델 변환기 60여 개 + 벤치마크 + 오라클
│   ├── scripts/              장시간 변환 헬퍼 (run.sh, supervisor.sh)
│   └── tests/            278개   의존성 없는 C/Python 테스트
├── web/                      React 18 + TypeScript + Vite + Tailwind 4
│   ├── src/App.tsx           채팅 UI
│   ├── src/Brain.tsx         ★ 19,456 전문가를 살아있는 피질로 시각화
│   ├── src/Profiling.tsx     턴별 시간 분해 (I/O 대기 / matmul / 어텐션 / LM head)
│   ├── src/lib/api.ts        순수 OpenAI API 클라이언트 (엔진 지식 0)
│   └── src/i18n/             en, it, de, zh-CN, zh-TW  ← ⚠️ 한국어 없음
├── desktop/                  Tauri v2 (Rust) 데스크톱 셸 — web/을 네이티브 창에 포장
├── docker/                   Dockerfile, Dockerfile.slim, docker-compose.yml
├── docs/                     45개 문서
│   ├── quickstart.md, api.md, benchmarks.md, tuning.md
│   ├── ENVIRONMENT.md (54 KB 환경변수 인벤토리), FORMATS.md (39 KB)
│   ├── cuda.md / metal.md / vulkan.md / windows.md
│   ├── api-reference/openapi.json   OpenAPI 스펙
│   └── experiments/          실측 실험 로그 (6×5090, 4×A6000, 연속 배칭 등)
├── .github/workflows/        ci.yml, check.yml, release.yml, site.yml
├── flake.nix                 Nix 재현 빌드
├── pyproject.toml            colibri-engine (console script: coli)
└── CHANGELOG.md              33 KB — 논문급 상세 변경 이력
```

### 코드 규모

| 언어 | 줄 수 |
|---|---|
| C | 86,596 |
| 헤더(.h) | 36,148 |
| CUDA(.cu) | 9,324 |
| Metal(.mm) | 3,150 |
| Python | 59,661 |
| TypeScript/TSX | 2,612 |
| Markdown | 15,068 |
| **합계** | **약 20만 줄** |

### 아키텍처 원칙

> **"모델 패밀리당 .c 하나, 공유 헤더 위에."** 엔진은 자기 아키텍처만 소유하고, 두 엔진이 모두 필요한 것(safetensors 리더, 컨테이너 디코더, 토크나이저, 전문가 캐시)은 **공유 헤더**에 둡니다. 한 엔진에만 들어간 수정이 형제 엔진에 도달하지 못하는 것이 이 프로젝트에서 반복되는 결함 유형이기 때문입니다.

---

## 3. 지원 모델 9종

| 모델 | 전체/활성 | 디스크 | RAM | GPU | 빌드 |
|---|---|---|---|---|---|
| **OLMoE** (AI2) | 7B / 1B | ~7 GB (int8) | **8 GB** | 불필요 | `make -C c olmoe` |
| **Qwen3.6-35B-A3B** (Alibaba) | 35B / 3B | ~20 GB (int4-gs64) | 24 GB | 선택 (2×8GB에서 1.44→**10.05 tok/s, 7.0배**, 출력 비트 동일) | `make -C c qwen36` |
| **Qwen3.8-Flash-Next** | 125B + 51B n-gram / 6B | ~185.5 GB (FP8) | 16 GB | CPU 전용 | `make -C c qwen38` |
| **DeepSeek V4 Flash** | 284B / 13B | ~167 GB (REAP 150B: ~85 GB) | 16 GB | 선택 (프리필 5~10배, 디코드 ~2.5배) | `make -C c deepseek-v4` |
| **GLM-5.3-Flash** (Z.ai, 비전) | 321B / 40B | ~195 GB | 25 GB | 불필요 | `make -C c glm53` |
| **DeepSeek V4.1 Flash** (비전) | 552B / 16B | ~510 GB | 16 GB | 불필요 | `make -C c deepseek_v41` |
| **GLM-5.2/5.3** ★기준 | 744B / 40B | ~372 GB | **16 GB** min / 24 GB 권장 | 불필요 | `make -C c glm` |
| **Inkling** (Thinking Machines) | 975B / 41B | ~469 GB | 25 GB (int4 dense) / ~120 GB (bf16) | 불필요 | `make -C c inkling` |
| **Kimi K3** (Moonshot) | **2.8T** / 104B | ~1.6 TB | 32 GB+ | 불필요 | `make -C c kimi_k3` |

> **GPU는 오직 더 빠르게만 만듭니다.** 속도는 디스크가 결정합니다 — 전문가를 디스크에서 스트리밍하기 때문입니다.

---

## 4. 실측 성능 (저장소 벤치마크)

| 하드웨어 | GLM-5.2 (744B) |
|---|---|
| 6× RTX 5090, 전문가 전량 상주 | **5.8–6.8 tok/s**, TTFT ~13초 |
| 128 GB CPU 전용 데스크톱 | ~1.8 tok/s (warm) |
| RTX 5070 Ti 노트북급 1장 | 1.07 tok/s |
| 25 GB RAM 개발 PC (프로젝트 출발점) | 0.05–0.1 tok/s (cold) |

품질도 측정합니다: Qwen3.6 int4-**gs64**는 per-row 대비 코사인 0.98777 → 0.99313, KL 0.109 → 0.080 (**양자화 오차 약 44% 감소**).

---

## 5. 설치 및 사용법

### 준비물

| | 최소 | 권장 |
|---|---|---|
| RAM | 16 GB (OLMoE는 8 GB) | 24 GB+ |
| 여유 디스크 | GLM-5.2용 약 380 GB | 빠른 NVMe SSD |
| OS | Linux / Windows 10·11 / macOS | 아무거나 |
| 도구 | Python 3 (+소스빌드 시 gcc·make·git) | — |
| GPU | **필요 없음** | 있으면 더 빠름 |

### 경로 A — 프리빌드 (권장, 컴파일러 불필요)

```bash
# https://github.com/JustVugg/colibri/releases 에서 플랫폼 아카이브 다운로드
mkdir colibri && tar xzf colibri-v1.11.0-linux-x86_64.tar.gz -C colibri && cd colibri
python3 coli info          # engine ready ✓
```
- Windows: `colibri-<version>-windows-x86_64.zip` 압축 해제 후 **`coli.cmd` 더블클릭**
- ⚠️ ARM64 Linux(Graviton/Ampere/라즈베리파이)는 배포본이 x86_64 전용 → 소스 빌드 필요

### 경로 B — 소스 빌드

```bash
sudo apt install -y build-essential git python3     # Linux
git clone https://github.com/JustVugg/colibri && cd colibri/c
./setup.sh                 # gcc/OpenMP 확인 → 빌드 → 자체 테스트
pip install -e .           # coli를 PATH에 등록 (선택)
```
- 바이너리를 다른 PC로 옮기면 `libgomp.so.1` 필요 → `sudo apt install -y libgomp1`
- `ARCH=native` 로 빌드하면 내 CPU의 벡터 명령을 최대한 사용

### 모델 받기

```bash
# 권장: 사전 변환된 gs64 + int8 MTP 컨테이너
# https://huggingface.co/mastouri/GLM-5.2-colibri-int4-g64-with-int8-mtp  (약 372 GB)

# 또는 FP8 원본에서 직접 변환 (샤드 단위, 재개 가능 — 756 GB 전체가 동시에 필요하지 않음)
./coli convert --model /nvme/glm52_i4
```

> ⚠️ **반드시 gs64 컨테이너**를 쓰세요. 구버전 per-row int4 미러(`mateogrgic/…`, `jlnsrk/…`)는 품질이 약 9pp 나쁘고, think 모드 무한 루프와 종료 실패의 근본 원인입니다(#455).
> ⚠️ MTP 헤드는 **int8**이어야 합니다 (int4면 draft 수락률 0%, #8). 확인: `ls -l <model>/out-mtp-*` → `3527131672 / 5366238584 / 1065950496`

### 실행 (처음이면 이 순서)

```bash
# 0) 가장 가벼운 OLMoE(7 GB)로 먼저 성공 경험 만들기
COLI_MODEL=/path/olmoe_i8 ./coli chat

# 1) 준비 상태 진단 (읽기 전용)
COLI_MODEL=/nvme/glm52_i4 ./coli doctor
COLI_MODEL=/nvme/glm52_i4 ./coli doctor --deep   # 텐서/샤드/인덱스/미러 정밀 검사

# 2) VRAM/RAM/디스크 배치 계획 미리보기
COLI_MODEL=/nvme/glm52_i4 ./coli plan

# 3) 이 머신의 가장 빠른 안전 프로파일 측정·저장
COLI_MODEL=/nvme/glm52_i4 ./coli tune

# 4) 대화
COLI_MODEL=/nvme/glm52_i4 ./coli chat

# 5) API 서버 + 웹 대시보드
./coli web   --model /nvme/glm52_i4     # 브라우저 자동 오픈
./coli serve --model /nvme/glm52_i4     # 헤드리스

# 6) 단발 실행
./coli run "안녕" --model /nvme/glm52_i4
```

`coli` 서브커맨드 전체: `build info plan mirror doctor tune run chat serve cluster stop web bench convert`

### 주요 옵션 / 환경변수

| 옵션 / 변수 | 뜻 |
|---|---|
| `COLI_MODEL=<dir>` / `--model` | 모델 디렉터리 (필수) |
| `--ram N` | RAM 예산(GB) → 전문가 캐시 자동 사이징 |
| `--cap N` | 레이어당 캐시 슬롯 (RAM 부족 시 `--cap 2`) |
| `--ngen N` | 최대 응답 토큰 |
| `--topk N` / `--topp P` | 고정 top-k / 적응형 전문가 top-p |
| `--repin N` | N 토큰마다 RAM/VRAM 전문가 재배치 |
| `COLI_MODEL_MIRROR=<dir>` | **두 번째 SSD에 모델 2벌 → 읽기 대역폭 2배** |
| `COLI_DISK_WEIGHTS=9,3` | 두 드라이브의 대역폭 비율 (기본: 시작 시 측정) |
| `COLI_API_KEY` | 서버 Bearer 토큰 (외부 노출 전 필수) |
| `COLI_ALLOWED_HOSTS` / `--allowed-host` | DNS 리바인딩 허용 호스트 (와일드카드 없음) |
| `--cors-origin` | 브라우저 origin 허용 |
| `--max-queue N` (기본 8) / `--queue-timeout` (기본 300s) | 요청 큐 |
| `COLI_K3_CKPT` / `COLI_K3_CKPT_DIR` | Kimi K3 재귀 상태 체크포인트 |
| `COLI_TOOL_SALVAGE=1` | 깨진 GLM int4 툴콜 복구 (opt-in) |

> 전체 인벤토리: `docs/ENVIRONMENT.md` (54 KB)

### Docker

```bash
MODEL_DIR=/nvme/glm52_i4 docker compose -f docker/docker-compose.yml up
curl http://localhost:5000/v1/chat/completions \
  -d '{"model":"glm-5.2","messages":[{"role":"user","content":"Hi"}]}'
```
모델은 호스트에서 **read-only 바인드 마운트**. NVMe/ext4에 두고 **네트워크 마운트는 절대 금지**(전문가 스트리밍은 레이턴시 바운드).

### 로컬 클러스터 (여러 대 묶기)

```bash
./coli cluster coordinator --host 0.0.0.0 --port 8765
./coli cluster worker --model /nvme/glm52_i4 --port 9100 \
  --coordinator http://COORDINATOR:8765 --advertise-host WORKER_IP
./coli serve --model /nvme/glm52_i4 --cluster-coordinator http://127.0.0.1:8765
```
코디네이터가 토큰 생성·라우팅·KV 상태를 로컬에 유지하고, 디스크 기반 전문가 워커가 다른 머신에서 라우팅된 FFN을 실행합니다. 레이어의 배치 유니온을 **하나의 영속 TCP 요청**으로 보내므로 전문가당 라운드트립이 발생하지 않습니다.

### 웹 UI 개발

```bash
cd web && npm install && npm run dev     # 기본 엔드포인트 http://127.0.0.1:8000/v1
npm test                                  # vitest (fetch 모킹)
npm run build                             # tsc -b && vite build
```

---

## 6. 플러그인? 스킬? MCP? → **셋 다 아님**

저장소 전체를 검색한 결과 **"MCP" / "Model Context Protocol" 문자열이 단 한 번도 등장하지 않습니다.** 플러그인 매니페스트도, 스킬 정의도 없습니다.

### 정확한 정체: **독립 실행형 추론 엔진 + API 서버**

llama.cpp, vLLM, Ollama와 **같은 계층**입니다. 다른 AI 호스트에 끼우는 확장이 아니라, **AI 모델을 실행하는 프로그램 그 자체**입니다.

| 구분 | colibrì |
|---|---|
| MCP 서버 (AI에게 도구를 붙이는 프로토콜) | ❌ |
| 플러그인 / 스킬 (기존 호스트의 확장) | ❌ |
| **추론 엔진 (모델을 실행하는 런타임)** | ✅ |

### 단, "MCP의 반대편"에 꽂을 수 있습니다

OpenAI 호환 API와 **Anthropic Messages API**를 동시에 서빙합니다.

| 엔드포인트 | 설명 |
|---|---|
| `GET /v1/models`, `/v1/models/{model}` | 모델 목록 |
| `POST /v1/chat/completions` | OpenAI 호환 — SSE 스트리밍, usage, max_tokens, temperature, top_p, stop(최대 4개) |
| `POST /v1/completions` | 레거시 |
| **`POST /v1/messages`** | **Anthropic Messages API 호환** (별도 활성화 불필요) |
| `GET /health` | active/queued/completed/rejected 카운터 |
| `GET /profile` | 턴별 성능 스냅샷 |

Claude Code를 로컬 모델로 돌리기:
```bash
export ANTHROPIC_BASE_URL=http://localhost:8000
export ANTHROPIC_API_KEY=local        # COLI_API_KEY를 설정했을 때만 실제로 검사됨
export ANTHROPIC_MODEL=glm-5.2-colibri
claude
```

툴 콜링 지원 매트릭스:

| 엔진 | OpenAI `tools` | Anthropic `tool_use` | 네이티브 포맷 |
|---|---|---|---|
| GLM-5.2 | ✅ | ✅ | `<tool_call>` 블록 |
| DeepSeek V4 / V4.1 | ✅ | ✅ | 네이티브 DSML |
| Kimi K3 | ✅ | ✅ | XTML `tools`/`call`/`argument` |
| Inkling / Qwen3.8 / OLMoE | ❌ | ❌ | HTTP 400 반환 |

> 💡 **기회**: colibrì용 MCP 서버는 아직 존재하지 않습니다. 만들면 사실상 최초입니다.

---

## 7. API 토큰이 필요한가

| 토큰 | 필요? | 비용 | 비고 |
|---|---|---|---|
| OpenAI API 키 | ❌ | — | 외부 서비스를 호출하지 않음 |
| Anthropic API 키 | ❌ | — | 더미 아무 값이면 됨 |
| **`COLI_API_KEY`** | ⚠️ 외부 노출 시 필수 | **무료** | 내가 정하는 Bearer 토큰 |
| **`HF_TOKEN`** / `COLI_HF_TOKEN` | 🔸 모델 다운로드/변환 시 | **무료** | gated 모델·속도 제한 회피 |

```bash
# 인증 켜기
COLI_API_KEY=내가정한비밀값 ./coli serve --model /nvme/glm52_i4

curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Authorization: Bearer 내가정한비밀값' \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.2-colibri","messages":[{"role":"user","content":"Hello"}]}'
```

> 🔒 문서의 명시적 경고: **기본 바인드는 localhost. 머신 밖으로 노출하기 전에 `COLI_API_KEY`를 설정하라.**
> 웹 UI는 API 키를 **메모리에만** 보관하며, 레거시 `colibri.apiKey` 저장값을 시작 시 제거합니다.

---

## 8. AI 에이전트 구축에 도움이 되는가 → **네, 용도를 고르면**

### 유리한 점

| 기능 | 에이전트에 왜 중요한가 |
|---|---|
| 툴 콜링 (GLM-5.2, DeepSeek V4/V4.1, Kimi K3) | ReAct·함수 호출 루프 구현 가능 |
| Anthropic `/v1/messages` 호환 | Claude SDK / Claude Code / LangChain 등이 코드 수정 없이 연결 |
| OpenAI `/v1/chat/completions` 호환 | 기존 에이전트 프레임워크 대부분 즉시 동작 |
| **GBNF 문법 강제 출력** (`grammar.h`, `schema_gbnf.h`) | JSON 파싱 실패를 문법 레벨에서 원천 차단 |
| **KV 프리픽스 재사용** (`kv_prefix.h`) | 긴 시스템 프롬프트를 턴마다 재계산하지 않음 |
| **영속 KV 컨텍스트** (`kv_persist.h`, `cache_slot`) | 세션 저장·복구, 웜 대화 유지 |
| Kimi K3 재귀 상태 체크포인트 | 프롬프트 수정 시 전체 재생 없이 꼬리만 재프리필 |
| MLA KV **57배** 압축 | 긴 컨텍스트 메모리 벽 완화 |
| 투기적 디코딩 (MTP / DSpark / grammar draft) | 속도 보강 (수락률이 검증 비용을 못 갚으면 비활성화 가능) |
| 비전 (GLM-5.3-Flash, DeepSeek V4.1, Qwen3.8) | 멀티모달 에이전트 |
| 라우팅 텔레메트리 (`route_trace.h`, `/profile`) | 워크로드별 전문가 사용 측정 → 튜닝 |
| **비용 0 / 무제한 토큰** | 에이전트는 토큰을 폭식함 — 로컬은 전기료만 |

### 치명적 제약

- **1~6 tok/s** → ReAct 10스텝 × 500토큰 = 수십 분. 실시간 대화형 에이전트 부적합
- **동시 생성 1개** (FIFO 큐, 기본 8, 포화 시 HTTP 429) → 멀티 에이전트 병렬 불가
- Inkling / Qwen3.8 / OLMoE는 툴 콜링 미지원
- 양자화 모델이 항상 유효한 툴 문법을 뱉지는 않음 (`COLI_TOOL_SALVAGE=1` 복구 경로 존재 = 깨지는 경우가 있다는 뜻)
- 이미지 / logprobs / 토큰 패널티 일부는 명시적 에러 반환

### 잘 맞는 에이전트

- ✅ **배치/야간 에이전트** — 대량 문서 요약, 레포 전체 코드 리뷰, 로그 분석
- ✅ **민감 데이터 에이전트** — 의료·법률·사내 기밀 (완전 오프라인)
- ✅ **비용 제약 실험** — 긴 컨텍스트를 수천 번 반복
- ✅ **하이브리드** (권장) — 빠른 응답/라우팅은 클라우드 API, 대량·민감·반복은 로컬 colibrì
- ❌ 실시간 챗봇, 고객 응대, 다중 사용자 서비스

### 실전 추천 조합

```
Qwen3.6-35B-A3B (20 GB, 2×8GB GPU에서 10 tok/s, 출력 비트 동일)
      ↓  에이전트 로직 개발·검증
GLM-5.2 744B (372 GB)  ← 품질이 필요한 최종 배치 실행
```

---

## 9. React / PHP로 만들 수 있는가

### 엔진 자체 → **불가능**

| 이유 | 내용 |
|---|---|
| 메모리 레벨 제어 | `mmap`, `O_DIRECT`, NUMA 바인딩, 페이지 캐시 우회 |
| SIMD / 벡터 명령 | AVX-512, NEON (`fused_simd.h`) |
| GPU 커널 | CUDA(.cu), Metal(.mm), Vulkan 셰이더 수천 줄 |
| 멀티스레딩 | OpenMP (`omp_tune.h`) |
| 양자화 비트 연산 | int4/fp4/fp8/mxfp4 (`quant.h` 2,102줄) |
| 성능 | 현재도 1~6 tok/s — JS/PHP면 수백~수천 배 느려 사실상 동작 불가 |

### 그런데 React는 **이미 쓰고 있습니다**

`web/` = React 18.3 + TypeScript 7 + Vite 8 + Tailwind 4 + vitest 4 (lucide-react, clsx, tailwind-merge, class-variance-authority).
`web/README.md` 원문: *"React/Vite interface for an **OpenAI-compatible** colibrì server."*
→ **UI는 엔진을 전혀 모르고 HTTP만 사용합니다.** `web/src/lib/api.ts`는 순수 OpenAI 클라이언트입니다.

### React로 할 수 있는 것

1. **한국어 i18n 추가** — `web/src/i18n/ko.ts` 생성 + `index.ts` 등록 (가장 쉽고 가치 높은 기여)
2. 완전히 새 UI (Next.js / Remix) — API가 OpenAI 호환이므로 자유
3. Brain / Atlas 시각화 개선 (Three.js, D3)
4. 에이전트 워크플로우 빌더 (React Flow 노드 에디터)
5. 모바일 리모트 컨트롤 (집 PC의 colibrì를 폰에서)
6. 멀티모델 관리 대시보드 (9개 패밀리 전환·모니터링)
7. Tauri 데스크톱 앱 확장 (`desktop/`에 Tauri v2 셸 이미 존재)

### PHP로 할 수 있는 것

```php
<?php
$ch = curl_init('http://127.0.0.1:8000/v1/chat/completions');
curl_setopt_array($ch, [
  CURLOPT_POST => true,
  CURLOPT_RETURNTRANSFER => true,
  CURLOPT_HTTPHEADER => [
    'Content-Type: application/json',
    'Authorization: Bearer ' . getenv('COLI_API_KEY'),
  ],
  CURLOPT_POSTFIELDS => json_encode([
    'model'    => 'glm-5.2-colibri',
    'messages' => [['role' => 'user', 'content' => $userInput]],
  ]),
]);
$res = json_decode(curl_exec($ch), true);
echo $res['choices'][0]['message']['content'];
```
- **WordPress 플러그인** — "내 서버의 로컬 AI로 글쓰기" (WP 점유율 40%+)
- **Laravel 사내 AI 포털** — 사번 로그인 + 부서 권한 + 사용 로그
- **그누보드/XE 연동** — 한국 시장 특화
- 기존 `openai-php/client` 류 SDK가 엔드포인트만 바꾸면 그대로 동작

### 레이어별 정리

| 레이어 | 언어 | React/PHP |
|---|---|---|
| 추론 엔진 (행렬연산·양자화·I/O) | C / CUDA / Metal / Vulkan | ❌ |
| 변환·플래너·게이트웨이 | Python | 🔸 가능하나 비권장 |
| **HTTP API (OpenAI/Anthropic 호환)** | — | ✅ **경계선, 여기서부터 자유** |
| 웹 UI / 대시보드 | React + TS (이미 존재) | ✅ |
| 서버 앱 / CMS 연동 | 무관 | ✅ PHP 완벽 가능 |

---

## 10. 유튜브 강의 영상 제작 가능성 → **최상급 소재**

| 요소 | 평가 |
|---|---|
| 후킹 | "월 0원으로 744B AI" / "GPU 없이 2.8조 파라미터" |
| 비주얼 | ⭐⭐⭐⭐⭐ Brain(19,456 전문가가 뇌처럼 번쩍), Atlas(3D 은하수) |
| 화제성 | ⭐40,376 / 🍴4,440, 2026-07 생성 — 3개월 만에 4만 스타 |
| 경쟁 | 한국어 콘텐츠 거의 없음 (i18n에 한국어조차 없음) = 블루오션 |
| 드라마 | 25 GB 노트북 0.05 tok/s → 6×5090 6.8 tok/s 스토리 |
| 분량 | 45개 문서 + 33 KB CHANGELOG + 실험 로그 → 20편 이상 |
| 라이선스 | Apache 2.0 — 영상 제작·수익화 자유 (고지 필요) |

### 추천 12편 시리즈

| # | 제목 | 핵심 장면 |
|---|---|---|
| 1 | GPU 없이 744B AI 돌리기 — 가능합니까? | 작업관리자 RAM 10 GB vs 모델 응답 |
| 2 | MoE가 뭔데? 19,456명의 요리사 비유 | 5.4% 활성화 애니메이션 |
| 3 | 설치 완전정복 (Win/Mac/Linux) | `coli.cmd` 더블클릭 무편집 |
| 4 | 모델 372 GB 받기: 함정 피하기 | gs64 vs per-row, int8 MTP (#455 무한루프 실화) |
| 5 | ⭐ Brain 페이지: AI의 뇌를 실시간으로 본다 | 전문가 흰색 번쩍 (최고 조회수 예상) |
| 6 | ⭐ Atlas: AI 안에 '시 담당'·'법률 담당'이 있다 | 3D 은하수, 주제별 클러스터 |
| 7 | 속도 올리기: tune/plan/doctor/듀얼 SSD | 대역폭 2배 before/after |
| 8 | CPU vs GPU 실측: 1.44 → 10.05 tok/s | Qwen3.6 + 8GB 2장, 출력 비트 동일 |
| 9 | 내 PC를 ChatGPT API로 (serve/web) | OpenAI SDK 몇 줄 연결 |
| 10 | ⭐ Claude Code를 로컬 AI로 돌리기 | `ANTHROPIC_BASE_URL` 3줄 |
| 11 | 로컬 AI 에이전트 (툴 콜링 + GBNF) | JSON 깨짐 0% 시연 |
| 12 | 9개 모델 비교: 7B ~ 2.8T | 요구사항 표 + 품질 비교 |

### 제작 수칙

- 1편은 **OLMoE(7 GB)** 로 성공 경험부터. 372 GB 다운로드로 시작하면 이탈
- 다운로드·변환은 타임랩스 + 자막 압축
- Brain/Atlas는 60fps 화면 녹화 → **쇼츠로 재활용** (15초 클립이 알고리즘에 강함)
- ✅ 속도(1~6 tok/s)와 디스크 372 GB 요구사항을 **영상 시작 30초 안에** 고지
- ✅ 벤치마크는 하드웨어·커밋·프롬프트·캐시 상태 명시 (프로젝트 문화 자체가 그렇습니다)
- ✅ Apache 2.0 고지 + 원 저작자 **Vincenzo Fornaro / JustVugg** 크레딧
- ❌ 모델 가중치 파일 직접 재배포 금지 (출처 링크만)

---

## 11. 수익화 아이디어 (상세)

> ⚖️ **라이선스**: 엔진은 **Apache 2.0** (Copyright 2026 Vincenzo Fornaro) — 상업적 이용 ✅ / 수정 ✅ / 재배포 ✅ / 특허 그랜트 ✅ / **소스 공개 의무 없음** ✅. 의무는 라이선스·NOTICE·저작권 고지 유지와 변경 표시.
> ⚠️ **모델 가중치는 별도 라이선스**입니다 (GLM-5.2 = Z.ai / MIT 등). 모델 포함 재판매 전 **각각 개별 확인 필수**.

### Tier 1 — 지금 바로, 자본 거의 0

#### 1. 한국어 생태계 선점 (⭐ 최우선)

**빈 틈**: `web/src/i18n/`에 `en, it, de, zh-CN, zh-TW` — **한국어 없음.** README도 영/중간체/중번체/이탈리아어만.

실행: `ko.ts` PR → `README.ko.md` PR → 한국어 설치 가이드 블로그 → 유튜브 시리즈 → 광고 + 제휴(SSD/RAM) + 유료 강의 + 컨설팅 문의
- 예상: 월 50~500만원 / 리스크: 낮음 / 투입: 시간

#### 2. 온프레미스 AI 구축 B2B (⭐ 수익성 최고)

가치 제안: *"귀사 데이터가 단 한 바이트도 외부로 나가지 않는 AI를, 구독료 0원으로."*

| 업종 | 니즈 |
|---|---|
| 병원·의료 | 환자기록 요약·검색 (개인정보보호법) |
| 법무법인 | 판례 검색, 계약서 검토 (수임 비밀) |
| 제조·방산 | 설계도면·공정 데이터 (기술 유출 금지) |
| 금융 | 내부 규정 Q&A (금융보안 규제) |
| 공공기관 | 망분리 필수 |
| 연구소 | 미공개 연구데이터 |

| 상품 | 가격(예시) |
|---|---|
| 진단 컨설팅 (적합성 분석 + `coli plan`/`doctor` 리포트) | 100~300만원 |
| 기본 구축 (하드웨어 선정·조립 + 변환 + 서빙 + 사내 UI) | 1,500~4,000만원 |
| 엔터프라이즈 (+ RAG, 권한, 로깅, 시스템 연동) | 5,000만원~ |
| 유지보수 MRR | 월 50~300만원 |

마진 근거: **소프트웨어 라이선스료 0원**, 하드웨어는 고객 자산, 내 인건비가 곧 매출.
⚠️ 리스크: **속도를 계약 전에 반드시 고지.** SLA를 토큰 속도가 아니라 **"처리 건수/일"** 로 작성.

#### 3. "로컬 AI 머신" 하드웨어 번들

근거: 속도 = SSD 성능이고, **듀얼 SSD 읽기 대역폭 2배**가 공식 지원(`COLI_MODEL_MIRROR`).

| 등급 | 구성 | 모델 | 가격대 |
|---|---|---|---|
| 입문 | 32 GB + 1 TB NVMe | Qwen3.6, OLMoE, DeepSeek V4 | 150~250만원 |
| 표준 | 64 GB + 2 TB NVMe ×2 | GLM-5.2 744B | 300~500만원 |
| 프로 | 128 GB + 4 TB ×2 + RTX 5090 | GLM-5.2 고속, Inkling | 800~1,500만원 |
| 극한 | 256 GB + 4 TB ×4 | Kimi K3 2.8T | 2,000만원~ |

차별점: **`coli tune` 최적화 완료 + 벤치마크 리포트 첨부 + 모델 사전 설치 + 한국어 UI**. 재고 리스크 회피를 위해 **주문 제작**으로 시작.

### Tier 2 — 개발 역량 필요

#### 4. colibrì MCP 서버 (오픈소스 선점, ⭐ 전략 가치 최고)
아직 존재하지 않음. 도구 예시: `colibri_chat`, `colibri_list_models`, `colibri_plan`, `colibri_doctor`, `colibri_expert_stats`, `colibri_profile`.
킬러 유스케이스: *민감 파일은 로컬 colibrì, 일반 작업은 클라우드* 자동 라우팅. 직접 수익은 작지만 평판·일감·강연 효과가 큼.

#### 5. WordPress 플러그인 "Local AI Writer" (PHP)
자체 서버 colibrì로 글 작성·요약·번역. 셀링 포인트 = **OpenAI 토큰 비용 0원**.
Freemium: 무료 / Pro 연 $49~99 (대량 배치, SEO 템플릿, 멀티사이트).
⚠️ 속도 → **백그라운드 큐 + 예약 발행** 구조로 설계하면 오히려 자연스러움.

#### 6. 한국형 CMS / 사내 포털 연동 (PHP·Laravel)
그누보드/XE/Laravel 사내 포털용 "로컬 AI Q&A" 모듈. 사번 로그인 + 부서 권한 + 감사 로그. 온프레미스 라이선스 또는 구축비.

#### 7. React 기반 "로컬 AI 에이전트 빌더"
React Flow 노드 에디터로 "PDF 읽기 → 요약 → 분류 → 엑셀 저장" 파이프라인 구성, 야간 배치 실행.
colibrì의 약점(느림)이 배치에서는 약점이 아니고, 강점(무제한 토큰·완전 프라이버시)만 남음. 오픈코어 모델.

#### 8. 성능 튜닝 컨설팅 (틈새, 고단가)
저장소의 "미해결 가설 표"가 곧 시장:
- 듀얼 SSD 콜드캐시 실측 A/B 부족
- 하드웨어 인식 플래너 vs 수동 파라미터 스윕 비교 필요
- MTP 투기 디코딩 손익분기 (85% 전문가 히트에서 **32% 손실** 측정)

도구: `coli tune`, `autotune.py`, `blocksweep.py`, 278개 테스트. 1건 300~1,000만원.
부수 효과: 측정 결과를 이슈로 공개 → 기여 + 전문성 증명 (프로젝트가 네거티브 결과도 환영).

### Tier 3 — 장기 / 대형

#### 9. 기업 연수 커리큘럼
MoE 구조, 양자화, KV 캐시, 메모리 계층, 추론 최적화를 **읽을 수 있는 C 코드**로 교육. 1인 50~150만원 × 20명.

#### 10. 틈새 어플라이언스
로펌 전용(판례 DB + 폐쇄망), 병원 전용(EMR 연동 의무기록 요약), 교육용 "AI 내부 관찰 키트"(Brain/Atlas가 교재로 우수).

#### 11. 모델 컨테이너 변환 서비스
근거: `c/tools/` 변환 스크립트 60여 개, 잘못된 컨테이너가 품질 9pp를 깎고 무한루프를 유발한 실제 사례(#455). 변환은 어렵고 틀리면 조용히 망가짐.
상품: "파인튜닝 모델을 colibrì 컨테이너로 변환 + 품질 검증 리포트(코사인/KL)".

### 우선순위 요약

| 순위 | 아이디어 | 초기비용 | 난이도 | 수익 잠재력 | 리스크 |
|---|---|---|---|---|---|
| 1 | 한국어 선점 (i18n + 콘텐츠 + 유튜브) | 거의 0 | ⭐⭐ | 중~상 | 낮음 |
| 2 | 온프레미스 구축 B2B | 낮음 | ⭐⭐⭐⭐ | 매우 높음 | 중 |
| 3 | MCP 서버 오픈소스 | 0 | ⭐⭐⭐ | 간접(평판) | 낮음 |
| 4 | 하드웨어 번들 | 높음 | ⭐⭐⭐ | 중~상 | 중 |
| 5 | WordPress 플러그인 | 낮음 | ⭐⭐⭐ | 중 | 중 |
| 6 | 에이전트 빌더 (React) | 낮음 | ⭐⭐⭐⭐ | 중~상 | 중 |
| 7 | 튜닝 컨설팅 | 0 | ⭐⭐⭐⭐⭐ | 중(고단가) | 낮음 |
| 8 | 교육/연수 | 낮음 | ⭐⭐⭐ | 중 | 낮음 |

### 90일 실행 플랜

- **1~30일**: `ko.ts` + `README.ko.md` PR → 한국어 설치 가이드 공개 → 유튜브 1~4편
- **31~60일**: 유튜브 5~8편 → 내 하드웨어 벤치마크를 이슈로 공개 → `colibri-mcp` v0.1 공개
- **61~90일**: 유입에서 B2B 문의 수집 → 진단 컨설팅(저가)으로 첫 매출 → 사례 1건 확보 후 구축 패키지 판매

### 공통 리스크

1. **속도 과장 금지** — 1~6 tok/s. 모든 상업 제안에 명시 (분쟁 1순위 원인)
2. **모델 라이선스 개별 확인** — 엔진은 Apache 2.0, 가중치는 각각 다름
3. **Apache 2.0 의무 이행** — LICENSE/NOTICE/저작권 고지 유지, 변경 표시
4. **프로젝트 자체 선언** — "속도 SLA 없음", 실험 기능 변동 가능 → 납품 시 **버전 고정(pin)**
5. **SSD 수명/부하** — 대량 읽기. 기업 납품은 엔터프라이즈·고 DWPD 드라이브 권장
6. **동시성 1** — 다중 사용자 서비스 불가. 큐·예약 구조로 설계
7. **시장 타이밍** — 하드웨어가 싸지면 가치 명제가 변할 수 있음 → 콘텐츠·컨설팅처럼 전환 가능한 자산에 투자

---

## 12. 종합 평가

### 강점
- 기술적 독창성이 뚜렷함 ("가중치를 위한 JIT", 메모리 다층화)
- **의존성 0** — 순수 C, BLAS 없음, 런타임 Python 없음, GPU 불필요
- 측정 문화 (가설 표, 실험 로그, 네거티브 결과 공개, 278개 테스트, torch/vLLM 오라클 대조)
- 모델 9종 지원 + OpenAI/Anthropic 이중 호환 + 웹/데스크톱/Docker/클러스터까지 완비
- Apache 2.0 — 상업 활용 자유
- 커뮤니티 모멘텀 (3개월 만에 ⭐4만)

### 약점·한계
- 느림 (1~6 tok/s), 동시 생성 1개
- 디스크 요구량 (372 GB ~ 1.6 TB)
- 모델 컨테이너 선택 실수의 함정이 존재 (gs64 / int8 MTP)
- "속도 SLA 없음" 공식 선언 — 실험적 성격
- 한국어 지원 전무 (← 동시에 가장 큰 기회)

### 결론
**속도를 사는 도구가 아니라, "가능성 · 소유권 · 학습"을 사는 도구.**
실시간 서비스용으로는 부적합하지만, **프라이버시가 필수인 온프레미스**, **대량 배치 처리**, **AI 시스템 학습·연구**, **콘텐츠 제작** 용도로는 현재 대안이 거의 없는 선택지입니다.
한국 시장에서는 **i18n·문서·콘텐츠가 모두 비어 있어 선점 가치가 큽니다.**

---

*이 문서는 저장소 전수조사(2026-10-08, 커밋 `2c2855e`, v1.11.0)를 기반으로 작성되었습니다.*
*원본 프로젝트: https://github.com/JustVugg/colibri — Apache License 2.0, Copyright 2026 Vincenzo Fornaro*
*분석 저장소: https://github.com/bmshin94/colibri*
