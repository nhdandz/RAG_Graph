import { Logo, LogoCreateRequest, LogoUpdateRequest, LogoResponse } from '../types/logo';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080/api';

export async function getLogos(): Promise<Logo[]> {
  const response = await fetch(`${API_BASE_URL}/logos`);
  if (!response.ok) {
    throw new Error('Failed to fetch logos');
  }
  return response.json();
}

export async function getAllLogos(): Promise<Logo[]> {
  const response = await fetch(`${API_BASE_URL}/admin/logos`);
  if (!response.ok) {
    throw new Error('Failed to fetch all logos');
  }
  return response.json();
}

export async function createLogo(data: LogoCreateRequest): Promise<LogoResponse> {
  const response = await fetch(`${API_BASE_URL}/admin/logos`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(data),
  });
  return response.json();
}

export async function updateLogo(id: string, data: LogoUpdateRequest): Promise<LogoResponse> {
  const response = await fetch(`${API_BASE_URL}/admin/logos/${id}`, {
    method: 'PUT',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(data),
  });
  return response.json();
}

export async function deleteLogo(id: string): Promise<LogoResponse> {
  const response = await fetch(`${API_BASE_URL}/admin/logos/${id}`, {
    method: 'DELETE',
  });
  return response.json();
}

export async function reorderLogos(logoIds: string[]): Promise<LogoResponse> {
  const response = await fetch(`${API_BASE_URL}/admin/logos/reorder`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ order: logoIds }),
  });
  return response.json();
}
