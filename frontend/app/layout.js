import "./globals.css";

export const metadata = {
  title: "Voice & Text Command Classifier",
  description:
    "AI-powered intent recognition for English and Nigerian Pidgin voice/text commands.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
