// Kraken Discord Bot - Custom Node Styling
// Applies the purple kraken theme to KrakenDiscordBot nodes
import { app } from "../../scripts/app.js";

// ── Config ─────────────────────────────────────────────────────────────────────
const NAME = "krakenDiscordBot.nodeStyle";
const IMAGE_FILE = "kraken_node_bg.png"; // place this file next to index.js
const HEADER_BG_HEX = "#28075B";         // purple title bar (header) color
const BODY_BG_HEX = "#200747";           // slightly darker body
const TITLE_TEXT_HEX = "#F0E8FF";        // title text color
const BG_OPACITY = 0.15;                 // wallpaper opacity (0..1)
const DEBUG_FAIL_COLOR = "#FF0000";      // Bright red to indicate image load failure

// ── Robust Image Loading ─────────────────────────────────────────────────────
const krakenBg = new Image();
krakenBg.src = new URL(`./${IMAGE_FILE}`, import.meta.url).href;

let bgReady = false;
let bgError = false;

krakenBg.onload = () => {
  if (krakenBg.naturalWidth > 0 && krakenBg.naturalHeight > 0) {
    bgReady = true;
    console.log(`[${NAME}] Theme wallpaper loaded successfully.`);
  } else {
    bgError = true;
  }
  try { app.graph?.setDirtyCanvas?.(true, true); } catch {}
};

krakenBg.onerror = () => {
  bgError = true;
  try { app.graph?.setDirtyCanvas?.(true, true); } catch {}
};

function canDrawBg() {
  return bgReady && !bgError;
}

// ── Styling Logic ──────────────────────────────────────────────────────────────

function isKrakenDiscordNode(node) {
  const id = (node?.comfyClass || node?.type || "").toString();
  // Match both KrakenDiscordBot and KrakenDiscordBotStatus
  return id.startsWith("KrakenDiscordBot");
}

function drawWallpaperClipped(ctx, w, h) {
  const LG = globalThis.LiteGraph;
  const TITLE_H = (LG && LG.NODE_TITLE_HEIGHT) || 20;

  ctx.save();
  ctx.beginPath();
  ctx.rect(0, TITLE_H, w, Math.max(0, h - TITLE_H));
  ctx.clip();

  ctx.globalAlpha = BG_OPACITY;
  ctx.drawImage(krakenBg, 0, TITLE_H, w, Math.max(0, h - TITLE_H));

  ctx.restore();
}

function applyStyle(node) {
  try {
    if (!isKrakenDiscordNode(node) || node._krakenStyled) return;

    node._krakenPrev = {
      color: node.color,
      bgcolor: node.bgcolor,
      title_color: node.title_color,
      onDrawBackground: node.onDrawBackground,
    };

    node.color = HEADER_BG_HEX;
    node.bgcolor = BODY_BG_HEX;
    node.title_color = TITLE_TEXT_HEX;

    const original_onDrawBackground = node.onDrawBackground;

    node.onDrawBackground = function (ctx) {
      original_onDrawBackground?.apply(this, arguments);

      if (canDrawBg() && this?.size) {
        drawWallpaperClipped(ctx, this.size[0], this.size[1]);
      } else if (bgError && this?.size) {
        // VISUAL DEBUG: If bgError is true, paint the body red.
        const LG = globalThis.LiteGraph;
        const TITLE_H = (LG && LG.NODE_TITLE_HEIGHT) || 20;
        ctx.fillStyle = DEBUG_FAIL_COLOR;
        ctx.fillRect(0, TITLE_H, this.size[0], this.size[1] - TITLE_H);
      }
    };

    node._krakenStyled = true;
    node.setDirtyCanvas?.(true, true);
  } catch (e) {
    console.error(`[${NAME}] Failed to apply style:`, e);
  }
}

app.registerExtension({
  name: NAME,
  nodeCreated(node) {
    setTimeout(() => applyStyle(node), 0);
  },
  loadedGraphNode(node) {
    applyStyle(node);
  },
});
