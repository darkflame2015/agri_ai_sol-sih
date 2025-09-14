export const API_BASE = (import.meta.env.VITE_API_BASE as string) || '/api';

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let msg = `HTTP ${res.status}`;
    try {
      const data = await res.json();
      if (data?.error) msg = data.error;
    } catch {}
    throw new Error(msg);
  }
  return res.json() as Promise<T>;
}

export async function createDevice(payload: { name: string; field: string }) {
  const res = await fetch(`${API_BASE}/devices`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  return handle<{ id: number; name: string; field: string }>(res);
}

export async function createField(payload: { name: string; cropType: string; area: number }) {
  const res = await fetch(`${API_BASE}/fields`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  return handle<{ id: number; name: string; cropType: string; area: number }>(res);
}
