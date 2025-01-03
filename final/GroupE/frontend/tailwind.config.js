const config = {
  darkMode: ["class"],

  content: {
    relative: true,
    files: [
      "./pages/**/*.{js,jsx}",
      "./components/**/*.{js,jsx}",
      "./app/**/*.{js,jsx}",
      "./src/**/*.{js,jsx}",
    ],
  },
  prefix: "",
  theme: {
    fontFamily: {
      sans: ["Inter", "sans-serif"],
    },
    container: {
      center: true,
      padding: "2rem",
      screens: {
        "2xl": "1400px",
      },
    },

    extend: {
      boxShadow: {
        tinyShadow:
          "1px 1px 4px 0px #0000000D, -1px -1px 4px 0px rgba(0, 0, 0, 0.05)",
        combinedShadow:
          "4px 4px 12px 0px #0000001A, -4px -4px 12px 0px #0000001A",
        overlayButtonShadow: "0px -1px 8px 8px #0000001A",
        columnShadow: "8px 0px 6px -1px rgb(251,245,255)",
      },

      colors: {
        error: "#E92E2E",
        primary: "#7743DB",
        secondary: "#B091EB",
        teritiary: "#F3ECFF",
        text_sub_base: "#887D9F",
        container: "#FBFAFC",
        button_Strok: "#E9E0EE",
        button_Column: "#FBF5FF",
        text_base: "#331966",
        placeholder: "#AFAEB0",
        primary_hover: "#8C60E1",
        click_delay: "#693DBC",
        background_column: "#F5F4F9",
        stroke_column: "#FBF5FF",
        success: "#1AC23F",
        // error: "#E92E2E",
        error_hover: "#F44343",
        error_clicked: "#CE2929",
        disabled: "#D1D1D1",
        animation: {
          "accordion-down": "accordion-down 0.2s ease-out",
          "accordion-up": "accordion-up 0.2s ease-out",
        },
      },

      keyframes: {
        "accordion-down": {
          from: { height: "0" },
          to: { height: "var(--radix-accordion-content-height)" },
        },
        "accordion-up": {
          from: { height: "var(--radix-accordion-content-height)" },
          to: { height: "0" },
        },
      },
      animation: {
        "accordion-down": "accordion-down 0.2s ease-out",
        "accordion-up": "accordion-up 0.2s ease-out",
      },
      fontFamily: {
        Inter: "Inter",
      },
    },
  },
  // plugins: [require("tailwindcss-animate")],
};

export default config;
