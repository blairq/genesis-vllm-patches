// Genesis: identifica la conversacion en cada pedido al vLLM propio (2026-10-05).
//
// Las cabeceras dejan que el servidor maneje el prefix cache por conversacion: desalojar primero lo de un
// subagente que termino, proteger al hilo principal frente a sus propios subagentes y bajarlo a RAM solo si
// entra otro hilo principal y falta VRAM. Solo van a los proveedores llm_saitama* (nada a terceros).
//
//   X-Genesis-Sesion  id de la sesion de opencode que hace el pedido
//   X-Genesis-Padre   sesion que la lanzo (vacio en un hilo principal)
//   X-Genesis-Raiz    hilo principal del arbol (la misma sesion si no tiene padre)
//   X-Genesis-Agente  agente de opencode (build, build_low, agi_explore, ...)
import type { Plugin } from "@opencode-ai/plugin";

// Proveedores que reciben las cabeceras (regex sobre el providerID de opencode); a ningun otro se le manda nada.
const PROVEEDOR = new RegExp(process.env.GENESIS_PROVEEDORES ?? "^llm_saitama");

const GenesisSesion: Plugin = async ({ client }) => {
  const padres = new Map<string, string>(); // sesion -> padre ("" si es raiz); el padre no cambia nunca

  async function padre(id: string): Promise<string> {
    const p = padres.get(id);
    if (p !== undefined) return p;
    try {
      const r = await client.session.get({ path: { id } });
      const v = r.data?.parentID ?? "";
      padres.set(id, v);
      return v;
    } catch {
      return ""; // sin el dato el pedido sale igual, solo sin padre
    }
  }

  async function raiz(id: string): Promise<string> {
    let r = id;
    for (let i = 0; i < 16; i++) {
      const p = await padre(r);
      if (!p) return r;
      r = p;
    }
    return r;
  }

  return {
    "chat.headers": async (input, output) => {
      if (!PROVEEDOR.test(input.model.providerID)) return;
      output.headers["X-Genesis-Sesion"] = input.sessionID;
      output.headers["X-Genesis-Padre"] = await padre(input.sessionID);
      output.headers["X-Genesis-Raiz"] = await raiz(input.sessionID);
      output.headers["X-Genesis-Agente"] = input.agent;
    },
  };
};

export default GenesisSesion;
