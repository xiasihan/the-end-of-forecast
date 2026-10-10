export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        // Base surfaces
        bg: '#f8f9fc',
        surface: '#ffffff',
        panel: '#f1f3f9',
        border: '#e5e7eb',
        borderStrong: '#d1d5db',

        // Text
        text: '#111827',
        textSecondary: '#4b5563',
        dim: '#9ca3af',

        // Accent - Indigo family
        accent: '#4f46e5',
        'accent-light': '#eef2ff',
        'accent-hover': '#4338ca',

        // Semantic
        green: '#10b981',
        'green-bg': '#ecfdf5',
        amber: '#f59e0b',
        'amber-bg': '#fffbeb',
        red: '#ef4444',
        'red-bg': '#fef2f2',
        blue: '#3b82f6',
        'blue-bg': '#eff6ff',
        purple: '#8b5cf6',
        'purple-bg': '#f5f3ff',
      },
      fontFamily: {
        sans: ['"DM Sans"', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', '"Fira Code"', 'monospace'],
      },
      boxShadow: {
        card: '0 1px 3px 0 rgba(0,0,0,0.04), 0 1px 2px -1px rgba(0,0,0,0.03)',
        popup: '0 20px 60px -12px rgba(0,0,0,0.12), 0 4px 24px -4px rgba(0,0,0,0.06)',
        glow: '0 0 0 3px rgba(79,70,229,0.12)',
      },
      borderRadius: {
        lg: '12px',
        xl: '16px',
        '2xl': '20px',
      },
    },
  },
  plugins: [],
};