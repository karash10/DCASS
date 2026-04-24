'use client';

import { useEffect, useMemo, useState } from 'react';
import Navigation from '@/components/Navigation';
import { Badge, Card, LoadingSpinner } from '@/components/UI';
import { DecodeResponse, decodeSequence } from '@/lib/api';

function parseMediaIds(input: string): string[] {
  return input
    .split(/[\s,]+/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function tokenizeForComparison(text: string): string[] {
  const stopwords = new Set([
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'have',
    'if', 'in', 'is', 'it', 'me', 'of', 'on', 'or', 'the', 'their', 'then',
    'they', 'to', 'we', 'when', 'with', 'you', 'your',
  ]);

  return text
    .toLowerCase()
    .match(/[a-z']+/g)?.filter((token) => !stopwords.has(token)) || [];
}

function compareMeaning(original: string, decoded: string) {
  const originalTokens = tokenizeForComparison(original);
  const decodedTokens = tokenizeForComparison(decoded);
  const decodedSet = new Set(decodedTokens);
  const overlap = Array.from(new Set(originalTokens.filter((token) => decodedSet.has(token))));
  const coverage = originalTokens.length ? overlap.length / originalTokens.length : 0;

  let label = 'Low';
  if (coverage >= 0.75) label = 'Strong';
  else if (coverage >= 0.4) label = 'Partial';

  return {
    originalTokens,
    decodedTokens,
    overlap,
    coverage,
    label,
    hasDirectOverlap: overlap.length > 0,
  };
}

export default function DecodePage() {
  const [mediaIdsText, setMediaIdsText] = useState('');
  const [originalMessage, setOriginalMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<DecodeResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (typeof window === 'undefined' || mediaIdsText) {
      return;
    }

    const params = new URLSearchParams(window.location.search);
    const ids = params.get('ids');
    const original = params.get('original');

    if (ids) {
      setMediaIdsText(ids.split(',').join('\n'));
    }
    if (original) {
      setOriginalMessage(original);
    }
  }, [mediaIdsText]);

  const parsedIds = useMemo(() => parseMediaIds(mediaIdsText), [mediaIdsText]);
  const comparison = useMemo(() => {
    if (!originalMessage.trim() || !result?.reconstructed_meaning) {
      return null;
    }
    return compareMeaning(originalMessage, result.reconstructed_meaning);
  }, [originalMessage, result?.reconstructed_meaning]);

  const handleDecode = async () => {
    if (!parsedIds.length) {
      setError('Please enter at least one media ID');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      const response = await decodeSequence({ media_ids: parsedIds });
      setResult(response);
    } catch (err: any) {
      console.error('Decoding error:', err);
      setError(err.response?.data?.detail || err.message || 'Decoding failed');
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  const handleUseExample = () => {
    setMediaIdsText('text_00001\nimage_00001');
    setResult(null);
    setError(null);
  };

  return (
    <>
      <Navigation />
      <main className="min-h-screen bg-background">
        <div className="container mx-auto px-4 py-8">
          <div className="flex items-center justify-between mb-8">
            <h1 className="text-3xl font-bold text-primary">Decode Sequence</h1>
            <Badge variant="info">Indices-Driven Decode</Badge>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="space-y-6">
              <Card title="Media IDs Input">
                <textarea
                  value={mediaIdsText}
                  onChange={(e) => setMediaIdsText(e.target.value)}
                  placeholder={'Paste media IDs here, one per line or comma-separated\n\nExample:\nimage_00123\ntext_00456'}
                  className="w-full h-72 bg-gray-800 border border-gray-700 rounded-lg p-4 text-white placeholder-gray-500 focus:outline-none focus:border-primary resize-none font-mono text-sm"
                />
                <div className="mt-3 flex items-center justify-between gap-4 text-sm">
                  <span className="text-gray-400">
                    {parsedIds.length} media ID{parsedIds.length === 1 ? '' : 's'} ready
                  </span>
                  <button
                    onClick={handleUseExample}
                    className="text-primary hover:text-primary/80"
                  >
                    Use sample IDs
                  </button>
                </div>
              </Card>

              <Card title="Original Message (Optional)">
                <textarea
                  value={originalMessage}
                  onChange={(e) => setOriginalMessage(e.target.value)}
                  placeholder="Paste the original message here to compare semantic recovery..."
                  className="w-full h-32 bg-gray-800 border border-gray-700 rounded-lg p-4 text-white placeholder-gray-500 focus:outline-none focus:border-primary resize-none"
                />
                <div className="mt-2 text-sm text-gray-400">
                  Add this if you want the page to estimate how much meaning was preserved.
                </div>
              </Card>

              <Card title="Decode Notes">
                <div className="space-y-3 text-sm text-gray-300">
                  <p>
                    This page decodes directly from the loaded FAISS index metadata. You do not need the original dataset files if your indices and metadata JSON files are present.
                  </p>
                  <p>
                    Unverified IDs will still be shown, so you can tell which packets are missing from the index.
                  </p>
                </div>
              </Card>

              <button
                onClick={handleDecode}
                disabled={loading || parsedIds.length === 0}
                className="w-full bg-primary hover:bg-primary/90 disabled:bg-gray-700 disabled:cursor-not-allowed text-white font-semibold py-4 px-6 rounded-lg transition-colors text-lg"
              >
                {loading ? '🔄 Decoding...' : '🔎 Decode Sequence'}
              </button>

              {error && (
                <div className="bg-error/20 border border-error/30 rounded-lg p-4 text-error">
                  ❌ {error}
                </div>
              )}
            </div>

            <div className="space-y-6">
              {loading && (
                <Card title="Decoding in Progress">
                  <LoadingSpinner />
                  <p className="text-center text-gray-400 mt-4">
                    Looking up media IDs from loaded index metadata...
                  </p>
                </Card>
              )}

              {result && !loading && (
                <>
                  <Card title="Decoded Meaning">
                    <div className="space-y-4">
                      <div className="text-gray-300 leading-7">
                        {result.reconstructed_meaning || 'No reconstructed meaning returned.'}
                      </div>
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div className="bg-gray-800 rounded-lg p-4">
                          <div className="text-sm text-gray-400 mb-1">Verification Rate</div>
                          <div className="text-2xl font-bold text-white">
                            {(result.verification_rate * 100).toFixed(1)}%
                          </div>
                        </div>
                        <div className="bg-gray-800 rounded-lg p-4">
                          <div className="text-sm text-gray-400 mb-1">All Verified</div>
                          <div className={`text-2xl font-bold ${result.all_verified ? 'text-success' : 'text-warning'}`}>
                            {result.all_verified ? 'Yes' : 'No'}
                          </div>
                        </div>
                        <div className="bg-gray-800 rounded-lg p-4">
                          <div className="text-sm text-gray-400 mb-1">Elapsed Time</div>
                          <div className="text-2xl font-bold text-white">
                            {result.elapsed_ms.toFixed(1)} ms
                          </div>
                        </div>
                      </div>
                    </div>
                  </Card>

                  {comparison && (
                    <Card title="Recovery Comparison">
                      <div className="space-y-5">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                          <div className="bg-gray-800 rounded-lg p-4">
                            <div className="text-sm text-gray-400 mb-2">Original Message</div>
                            <div className="text-gray-200 leading-7">{originalMessage}</div>
                          </div>
                          <div className="bg-gray-800 rounded-lg p-4">
                            <div className="text-sm text-gray-400 mb-2">Decoded Meaning</div>
                            <div className="text-gray-200 leading-7">{result.reconstructed_meaning}</div>
                          </div>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                          <div className="bg-gray-800 rounded-lg p-4">
                            <div className="text-sm text-gray-400 mb-1">Recovery Score</div>
                            {comparison.hasDirectOverlap ? (
                              <div className="text-2xl font-bold text-white">
                                {(comparison.coverage * 100).toFixed(1)}%
                              </div>
                            ) : (
                              <div className="text-xl font-bold text-gray-400">
                                Hidden
                              </div>
                            )}
                          </div>
                          <div className="bg-gray-800 rounded-lg p-4">
                            <div className="text-sm text-gray-400 mb-1">Recovery Label</div>
                            <div className={`text-2xl font-bold ${
                              !comparison.hasDirectOverlap
                                ? 'text-gray-400'
                                : comparison.label === 'Strong'
                                ? 'text-success'
                                : comparison.label === 'Partial'
                                  ? 'text-warning'
                                  : 'text-error'
                            }`}>
                              {comparison.hasDirectOverlap ? comparison.label : 'Needs Review'}
                            </div>
                          </div>
                          <div className="bg-gray-800 rounded-lg p-4">
                            <div className="text-sm text-gray-400 mb-1">Matched Keywords</div>
                            <div className="text-2xl font-bold text-white">
                              {comparison.overlap.length}/{comparison.originalTokens.length}
                            </div>
                          </div>
                        </div>

                        <div className="bg-gray-800 rounded-lg p-4">
                          <div className="text-sm text-gray-400 mb-2">Shared Keywords</div>
                          <div className="flex flex-wrap gap-2">
                            {comparison.overlap.length > 0 ? (
                              comparison.overlap.map((token) => (
                                <Badge key={token} variant="success">{token}</Badge>
                              ))
                            ) : (
                              <span className="text-gray-500">No direct keyword overlap found.</span>
                            )}
                          </div>
                        </div>

                        <div className="text-xs text-gray-500 leading-6">
                          This is an approximate semantic recovery view based on keyword overlap. When there is no direct keyword overlap, the numeric score is hidden because a literal 0 can be misleading for paraphrased outputs.
                        </div>
                      </div>
                    </Card>
                  )}

                  <Card title="Decoded Items">
                    <div className="space-y-3 max-h-[520px] overflow-y-auto">
                      {result.items.map((item, index) => (
                        <div key={`${item.media_id}-${index}`} className="bg-gray-800 rounded-lg p-4 border border-gray-700">
                          <div className="flex items-start justify-between gap-3 mb-2">
                            <div>
                              <div className="text-xs text-gray-500 mb-1">Item {index + 1}</div>
                              <div className="font-mono text-sm text-primary break-all">{item.media_id}</div>
                            </div>
                            <div className="flex items-center gap-2">
                              <Badge variant={item.verified ? 'success' : 'warning'}>
                                {item.verified ? 'Verified' : 'Unverified'}
                              </Badge>
                              <Badge variant="info">{item.modality}</Badge>
                            </div>
                          </div>
                          <div className="text-sm text-gray-300 leading-6">
                            {item.content || 'No content available'}
                          </div>
                        </div>
                      ))}
                    </div>
                  </Card>
                </>
              )}

              {!result && !loading && (
                <Card title="Decode Results">
                  <div className="text-center text-gray-500 py-12">
                    Paste media IDs from the encode flow or wire output, then run decode.
                  </div>
                </Card>
              )}
            </div>
          </div>
        </div>
      </main>
    </>
  );
}
