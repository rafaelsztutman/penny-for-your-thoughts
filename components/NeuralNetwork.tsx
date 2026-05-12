"use client";

import { useEffect, useRef } from "react";

type Node = {
  x: number;
  y: number;
  layer: number;
  idx: number;
  phase: number;
  speed: number;
  radius: number;
  activation: number;
};

type Edge = {
  a: Node;
  b: Node;
  weight: number;
  phase: number;
};

type Pulse = {
  edge: Edge;
  t: number;
  life: number;
};

type Props = {
  activity?: number;
  accent?: string;
  height?: string;
};

/**
 * Persistent neural-network canvas visualization.
 * Nodes pulse, edges flicker. When `activity` rises above 0.5 a cascade wave
 * travels left→right and lights up nodes/edges in the accent color.
 */
export function NeuralNetwork({
  activity = 0,
  accent = "#c97548",
  height = "100%",
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const activityRef = useRef(activity);
  activityRef.current = activity;
  const accentRef = useRef(accent);
  accentRef.current = accent;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let raf = 0;
    let w = 0;
    let h = 0;
    let dpr = 1;
    let nodes: Node[] = [];
    let edges: Edge[] = [];
    let pulses: Pulse[] = [];

    const hexToRgb = (hex: string): [number, number, number] => {
      const m = hex.replace("#", "");
      const x = m.length === 3 ? m.replace(/./g, (c) => c + c) : m;
      const n = parseInt(x, 16);
      return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
    };

    function build() {
      const layers = 7;
      const minNodes = 5;
      const maxNodes = 10;
      nodes = [];
      edges = [];
      const margin = { x: w * 0.08, y: h * 0.12 };
      const layerW = (w - margin.x * 2) / (layers - 1);

      for (let L = 0; L < layers; L++) {
        const count = Math.round(
          minNodes +
            (maxNodes - minNodes) *
              (1 - Math.abs(L - layers / 2) / (layers / 2)) *
              0.9,
        );
        const layerH = h - margin.y * 2;
        for (let i = 0; i < count; i++) {
          const t = count === 1 ? 0.5 : i / (count - 1);
          const jitter = Math.sin(L * 13 + i * 7) * 0.5 * 12;
          nodes.push({
            x: margin.x + L * layerW + Math.sin(i * 3 + L) * 6,
            y: margin.y + t * layerH + jitter,
            layer: L,
            idx: i,
            phase: Math.random() * Math.PI * 2,
            speed: 0.4 + Math.random() * 0.6,
            radius: 1.6 + Math.random() * 1.4,
            activation: 0,
          });
        }
      }

      for (let i = 0; i < nodes.length; i++) {
        const a = nodes[i];
        const nextLayer = nodes.filter((n) => n.layer === a.layer + 1);
        nextLayer.forEach((b) => {
          const dist = Math.hypot(b.x - a.x, b.y - a.y);
          if (Math.random() < 0.55 - dist / (w * 1.4)) {
            edges.push({
              a,
              b,
              weight: 0.15 + Math.random() * 0.85,
              phase: Math.random() * Math.PI * 2,
            });
          }
        });
      }
    }

    function resize() {
      dpr = window.devicePixelRatio || 1;
      const rect = canvas!.getBoundingClientRect();
      w = rect.width;
      h = rect.height;
      canvas!.width = w * dpr;
      canvas!.height = h * dpr;
      ctx!.setTransform(dpr, 0, 0, dpr, 0, 0);
      build();
    }

    let cascadeT = -1;
    let lastActivity = 0;

    function tick(t: number) {
      const time = t / 1000;
      const act = activityRef.current;
      if (act > 0.5 && lastActivity <= 0.5) cascadeT = time;
      lastActivity = act;

      // Trail-fade toward the cream page background.
      ctx!.fillStyle = "rgba(252, 248, 240, 0.20)";
      ctx!.fillRect(0, 0, w, h);

      const [ar, ag, ab] = hexToRgb(accentRef.current);
      // Inactive edges/nodes need a darker neutral on a light bg.
      const baseEdge = `rgba(90, 70, 50, `;
      const accentEdge = `rgba(${ar}, ${ag}, ${ab}, `;

      const cascadeAge = cascadeT < 0 ? 999 : time - cascadeT;
      const cascadeX = cascadeAge < 2.5 ? (cascadeAge / 2.5) * w : -1;

      nodes.forEach((n) => {
        const distFromWave = cascadeX < 0 ? 999 : Math.abs(n.x - cascadeX);
        const waveEnergy =
          distFromWave < w * 0.18
            ? Math.exp(-Math.pow(distFromWave / (w * 0.06), 2))
            : 0;
        n.activation = Math.max(n.activation * 0.94, waveEnergy);
      });

      edges.forEach((e) => {
        const avgAct = (e.a.activation + e.b.activation) * 0.5;
        const flicker =
          0.35 + 0.65 * (Math.sin(time * 2 * e.weight + e.phase) * 0.5 + 0.5);
        const baseAlpha = 0.05 + 0.08 * e.weight * flicker;
        const activeAlpha = 0.7 * avgAct * e.weight;

        ctx!.strokeStyle = baseEdge + baseAlpha * (1 + act * 0.4) + ")";
        ctx!.lineWidth = 0.5;
        ctx!.beginPath();
        ctx!.moveTo(e.a.x, e.a.y);
        ctx!.lineTo(e.b.x, e.b.y);
        ctx!.stroke();

        if (activeAlpha > 0.02) {
          ctx!.strokeStyle = accentEdge + activeAlpha + ")";
          ctx!.lineWidth = 0.8 + e.weight * 0.6;
          ctx!.beginPath();
          ctx!.moveTo(e.a.x, e.a.y);
          ctx!.lineTo(e.b.x, e.b.y);
          ctx!.stroke();
        }
      });

      const pulseRate = 0.04 + act * 0.25;
      if (Math.random() < pulseRate && edges.length) {
        const e = edges[(Math.random() * edges.length) | 0];
        pulses.push({ edge: e, t: 0, life: 1.2 + Math.random() * 0.8 });
      }
      pulses = pulses.filter((p) => {
        p.t += 0.016 / p.life;
        if (p.t >= 1) return false;
        const x = p.edge.a.x + (p.edge.b.x - p.edge.a.x) * p.t;
        const y = p.edge.a.y + (p.edge.b.y - p.edge.a.y) * p.t;
        const isActive = p.edge.a.activation + p.edge.b.activation > 0.2;
        const color = isActive ? accentEdge : "rgba(100, 80, 60, ";
        const alpha = Math.sin(p.t * Math.PI) * (isActive ? 0.9 : 0.45);
        ctx!.fillStyle = color + alpha + ")";
        ctx!.beginPath();
        ctx!.arc(x, y, isActive ? 1.6 : 1.0, 0, Math.PI * 2);
        ctx!.fill();
        return true;
      });

      nodes.forEach((n) => {
        const pulse = Math.sin(time * n.speed + n.phase) * 0.5 + 0.5;
        const baseR = n.radius * (0.85 + pulse * 0.3);
        const activeBoost = n.activation;

        if (activeBoost > 0.05) {
          const glow = ctx!.createRadialGradient(n.x, n.y, 0, n.x, n.y, baseR * 8);
          glow.addColorStop(0, accentEdge + activeBoost * 0.5 + ")");
          glow.addColorStop(1, accentEdge + "0)");
          ctx!.fillStyle = glow;
          ctx!.beginPath();
          ctx!.arc(n.x, n.y, baseR * 8, 0, Math.PI * 2);
          ctx!.fill();
        }

        const bodyAlpha = 0.35 + pulse * 0.25 + activeBoost * 0.5;
        ctx!.fillStyle =
          activeBoost > 0.1
            ? accentEdge + bodyAlpha + ")"
            : `rgba(80, 60, 45, ${bodyAlpha})`;
        ctx!.beginPath();
        ctx!.arc(n.x, n.y, baseR + activeBoost * 1.2, 0, Math.PI * 2);
        ctx!.fill();
      });

      raf = requestAnimationFrame(tick);
    }

    resize();
    raf = requestAnimationFrame(tick);
    const ro = new ResizeObserver(resize);
    ro.observe(canvas);
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
    };
  }, []);

  return (
    <canvas ref={canvasRef} style={{ width: "100%", height, display: "block" }} />
  );
}
