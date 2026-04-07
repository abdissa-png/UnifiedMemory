// @ts-check
/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Unified Memory System',
  tagline: 'MDX deep-dive documentation',
  favicon: 'img/favicon.svg',
  url: 'https://example.com',
  baseUrl: '/',
  organizationName: 'unified-memory',
  projectName: 'memory_system',
  onBrokenLinks: 'warn',
  i18n: { defaultLocale: 'en', locales: ['en'] },
  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          path: '../mdx',
          routeBasePath: 'docs',
          sidebarPath: './sidebars.js',
          /** Do not treat the folder README as a doc page */
          exclude: ['README.md'],
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],
  themes: ['@docusaurus/theme-mermaid'],
  markdown: {
    mermaid: true,
    hooks: {
      onBrokenMarkdownLinks: 'warn',
    },
  },
  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      colorMode: { respectPrefersColorScheme: true },
      navbar: {
        title: 'Unified Memory',
        items: [
          {
            type: 'doc',
            docId: 'system-overview',
            label: 'Deep dive',
            position: 'left',
          },
        ],
      },
      footer: {
        style: 'dark',
        copyright: `Copyright © ${new Date().getFullYear()} Unified Memory System.`,
      },
    }),
};

module.exports = config;
