import { Plugin } from "@opencode/plugin";

// Padrões de arquivos de credenciais/segredos que nunca devem ser lidos pelo LLM.
const BLOCKED_READ_PATTERNS: RegExp[] = [
  /(^|\/)\.env(\.|$)/,
  /(^|\/)\.envrc$/,
  /service[-_]account.*\.json$/i,
  /articulate-case-.*\.json$/i,
  /\.(pem|p12|pfx|key)$/i,
];

// Exceções permitidas (ex.: template de variáveis sem valores reais).
const ALLOWLIST: RegExp[] = [/(^|\/)\.env\.example$/];

export default Plugin.define({
  id: "env-protection",
  async setup(ctx) {
    // DIAGNÓSTICO TEMPORÁRIO
    const { writeFileSync } = await import("node:fs");
    const tools = await ctx.tool.list();
    writeFileSync(
      "/tmp/opencode/plugin-tools.log",
      tools.map((t) => t.id).sort().join("\n"),
    );
    // Intercepta cada execução de ferramenta antes que ela ocorra.
    await ctx.tool.hook("execute.before", (event) => {
      writeFileSync(
        "/tmp/opencode/plugin-hook.log",
        `${event.tool} ${JSON.stringify(event.input)}\n`,
        { flag: "a" },
      );
      // Avalia apenas a ferramenta de leitura de arquivos.
      if (event.tool !== "read") return;

      const filePath = (event.input as { filePath?: string } | undefined)
        ?.filePath;
      if (!filePath) return;

      const normalized = filePath.replace(/\\/g, "/");
      if (ALLOWLIST.some((re) => re.test(normalized))) return;
      if (BLOCKED_READ_PATTERNS.some((re) => re.test(normalized))) {
        throw new Error(
          "Leitura bloqueada: arquivo de credenciais/segredos. Acesse apenas via variáveis de ambiente.",
        );
      }
    });
  },
});
