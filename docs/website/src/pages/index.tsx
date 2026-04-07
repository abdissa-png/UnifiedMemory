import React from 'react';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';

/** Slugs match Docusaurus doc IDs (NN- prefix stripped from filenames). */
const chapters = [
  { id: 'system-overview', title: 'System overview' },
  { id: 'bootstrap-and-providers', title: 'Bootstrap and providers' },
  { id: 'ingestion-pipeline', title: 'Ingestion pipeline' },
  { id: 'retrieval-fusion-rerank', title: 'Retrieval, fusion, reranking' },
  { id: 'storage-data-plane', title: 'Storage and data plane' },
  { id: 'api-tenants-auth-observability', title: 'API, tenants, auth, observability' },
  { id: 'agents-workflows-and-apps', title: 'Agents, workflows, and client apps' },
  { id: 'domain-validation-and-quality', title: 'Domain validation and quality' },
  { id: 'configuration-matrix', title: 'Configuration matrix' },
];

export default function Home() {
  return (
    <Layout title="Home" description="Unified Memory System — MDX deep-dive docs">
      <main style={{ maxWidth: 720, margin: '0 auto', padding: '2rem 1.5rem' }}>
        <h1>Unified Memory System</h1>
        <p>
          Extended diagram-heavy documentation (MDX + Mermaid). Canonical Markdown guides live in the
          repository <code>docs/</code> folder; this site renders the <code>docs/mdx/</code>{' '}
          chapters.
        </p>
        <h2>Chapters</h2>
        <ul>
          {chapters.map((c) => (
            <li key={c.id}>
              <Link to={`/docs/${c.id}`}>{c.title}</Link>
            </li>
          ))}
        </ul>
        <p>
          <Link to="/docs/system-overview">Start with system overview →</Link>
        </p>
      </main>
    </Layout>
  );
}
