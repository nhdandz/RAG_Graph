export interface Logo {
  id: string;
  name: string;
  imageUrl: string;
  websiteUrl?: string;
  order: number;
  active: boolean;
  createdAt?: string;
}

export interface LogoCreateRequest {
  name: string;
  imageUrl: string;
  websiteUrl?: string;
  order: number;
}

export interface LogoUpdateRequest {
  name?: string;
  imageUrl?: string;
  websiteUrl?: string;
  order?: number;
  active?: boolean;
}

export interface LogoResponse {
  success: boolean;
  logo?: Logo;
  logos?: Logo[];
  message?: string;
  error?: string;
}
