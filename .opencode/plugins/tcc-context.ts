import { Plugin } from "@opencode/plugin";

// Contexto científico e metodológico que deve sobreviver à compactação de sessão.
const TCC_CONTEXT = [
  "## Contexto do projeto (TCC)",
  "- Pesquisa: segmentação semântica de cafezais (Região Geográfica Imediata de Guaxupé-MG) comparando U-Net (CNN) e SegFormer (ViT).",
  "- Fonte espectral: Sentinel-2 Level-2A, bandas B2/B3/B4/B8 a 10 m, patches 512x512.",
  "- Loss multivariada: Dice + Focal + Boundary. Validação: k-fold espacial (k=5).",
  "- Métricas pixel-level: IoU, F1-Score, Precision, Recall. XAI: Grad-CAM (U-Net) e Attention Rollout (SegFormer).",
  "- Regras de escrita: markdown de notebook em pt-br impessoal; identificadores en-US; uma responsabilidade por célula;",
  "  caminhos sempre via src/config.py (fonte única src/config.yaml); notebooks seguem a ordem do PLAN.md.",
  "- Notebooks são agnósticos de plataforma: o mesmo .ipynb roda corretamente no Google Colab e no",
  "  Kaggle (mesmo protocolo de processamento, sem exigir saída bit-a-bit entre hardwares);",
  "  detecção de ambiente, montagem do Drive e resolução de caminhos ficam apenas em src/.",
  "- Armazenamento canônico: todo dado/arquivo/artefato é gravado dentro da pasta tcc/ na raiz do Google Drive",
  "  (MyDrive/tcc/), acessada se já existir ou criada (com todas as subpastas) pelos próprios notebooks.",
  "- Segredos apenas por variáveis de ambiente; nunca versionar, logar ou expor credenciais.",
  "- Autenticação (GEE OAuth com conta principal via ee.Authenticate(), Hugging Face, Kaggle) executada",
  "  dentro dos próprios notebooks, nas células de código, com as bibliotecas necessárias — sem etapa",
  "  externa ou manual de autenticação.",
  "- Boas práticas de engenharia em todo o código (incl. células dos notebooks): DRY, responsabilidade única,",
  "  KISS, YAGNI e separação de conceitos — notebooks orquestram/visualizam, sem duplicar a lógica de src/.",
].join("\n");

export default Plugin.define({
  id: "tcc-context",
  async setup(ctx) {
    // Injeta o contexto do projeto nas instruções de sistema da compactação de sessão.
    await ctx.session.hook("compaction", (event) => {
      event.system.push({ type: "text", text: TCC_CONTEXT });
    });
  },
});
