import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  images: {
    remotePatterns: [
      {
        protocol: 'https',
        hostname: 'v0.dev',
        port: '', // Leave empty if no specific port is required
        pathname: '/**', // Matches all paths under the hostname
      },
    ],
  },
};

export default nextConfig;
