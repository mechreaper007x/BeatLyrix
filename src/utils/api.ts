// Centralized API Base URL resolver for BeatLyrix microservices

export const getApiUrl = (path: string): string => {
  const env = (import.meta as any).env || {};
  const rawApiUrl = env.VITE_API_URL || "https://raprank-backend.onrender.com";
  const baseUrl = rawApiUrl.replace(/\/$/, "");
  const cleanPath = path.startsWith("/") ? path : `/${path}`;
  return `${baseUrl}${cleanPath}`;
};

export const getUploadApiUrl = (path: string): string => {
  const env = (import.meta as any).env || {};
  const rawUploadUrl = env.VITE_UPLOAD_API_URL || "https://raprank-upload.onrender.com";
  const baseUrl = rawUploadUrl.replace(/\/$/, "");
  const cleanPath = path.startsWith("/") ? path : `/${path}`;
  return `${baseUrl}${cleanPath}`;
};
