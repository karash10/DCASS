'use client';

import { useEffect, useState } from 'react';
import Navigation from '@/components/Navigation';
import { Card, Badge } from '@/components/UI';
import axios from 'axios';

interface Packet {
  media_id: string;
  channel_id: number;
  sequence_number: number;
  delay_seconds: number;
  timestamp: number;
  mode_used: string;
  filename: string;
}

export default function WirePage() {
  const [packets, setPackets] = useState<Packet[]>([]);
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [stats, setStats] = useState({
    totalPackets: 0,
    avgDelay: 0,
    channelsUsed: new Set<number>(),
    modeUsed: 'unknown' as string,
  });

  useEffect(() => {
    // Poll shared_channel directory for packets
    const pollPackets = async () => {
      try {
        // This is a mock - in production, you'd set up a proper file watcher
        // or WebSocket connection. For now, we'll show example functionality.
        
        // We can't directly access file system from browser, so this would need
        // a backend API endpoint that lists files in shared_channel/
        
        // For demonstration, we'll use a mock data approach
        console.log('Monitoring shared_channel...');
      } catch (err) {
        console.error('Error polling packets:', err);
      }
    };

    let interval: NodeJS.Timeout;
    if (isMonitoring) {
      interval = setInterval(pollPackets, 1000);
    }

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isMonitoring]);

  // Update stats when packets change
  useEffect(() => {
    if (packets.length > 0) {
      const totalDelay = packets.reduce((sum, p) => sum + p.delay_seconds, 0);
      const channels = new Set(packets.map(p => p.channel_id));
      const lastMode = packets[packets.length - 1]?.mode_used || 'unknown';

      setStats({
        totalPackets: packets.length,
        avgDelay: totalDelay / packets.length,
        channelsUsed: channels,
        modeUsed: lastMode,
      });
    }
  }, [packets]);

  const getModeColor = (mode: string) => {
    switch (mode.toLowerCase()) {
      case 'rl': return 'success';
      case 'gan': return 'info';
      case 'static': return 'warning';
      default: return 'error';
    }
  };

  const clearPackets = () => {
    setPackets([]);
  };

  return (
    <>
      <Navigation />
      <main className="min-h-screen bg-background">
        <div className="container mx-auto px-4 py-8">
          <div className="flex items-center justify-between mb-8">
            <h1 className="text-3xl font-bold text-primary">Wire View</h1>
            <div className="flex items-center space-x-4">
              <Badge variant={stats.modeUsed === 'static' ? 'warning' : 'success'}>
                Mode: {stats.modeUsed.toUpperCase()}
              </Badge>
              <button
                onClick={() => setIsMonitoring(!isMonitoring)}
                className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                  isMonitoring
                    ? 'bg-error hover:bg-error/90 text-white'
                    : 'bg-success hover:bg-success/90 text-white'
                }`}
              >
                {isMonitoring ? '⏸️ Stop Monitoring' : '▶️ Start Monitoring'}
              </button>
            </div>
          </div>

          {/* Statistics */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
            <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
              <div className="text-sm text-gray-400 mb-1">Total Packets</div>
              <div className="text-2xl font-bold text-primary">{stats.totalPackets}</div>
            </div>
            <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
              <div className="text-sm text-gray-400 mb-1">Avg Delay</div>
              <div className="text-2xl font-bold text-white">
                {stats.avgDelay > 0 ? `${stats.avgDelay.toFixed(2)}s` : '--'}
              </div>
            </div>
            <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
              <div className="text-sm text-gray-400 mb-1">Channels Used</div>
              <div className="text-2xl font-bold text-white">{stats.channelsUsed.size}</div>
            </div>
            <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
              <div className="text-sm text-gray-400 mb-1">Active Mode</div>
              <div className="text-xl font-bold">
                <Badge variant={getModeColor(stats.modeUsed)}>
                  {stats.modeUsed.toUpperCase()}
                </Badge>
              </div>
            </div>
          </div>

          {/* Live Feed */}
          <Card title="Live Transmission Feed">
            <div className="mb-4 flex justify-between items-center">
              <div className="text-sm text-gray-400">
                {isMonitoring ? (
                  <span className="flex items-center">
                    <span className="animate-pulse mr-2">🔴</span>
                    Monitoring shared_channel/
                  </span>
                ) : (
                  <span className="text-gray-500">Monitoring paused</span>
                )}
              </div>
              {packets.length > 0 && (
                <button
                  onClick={clearPackets}
                  className="text-sm text-error hover:text-error/80"
                >
                  Clear All
                </button>
              )}
            </div>

            <div className="space-y-2 max-h-[600px] overflow-y-auto">
              {packets.length === 0 ? (
                <div className="text-center py-12 text-gray-500">
                  <div className="text-4xl mb-4">📡</div>
                  <p>No packets yet</p>
                  <p className="text-sm mt-2">
                    {isMonitoring 
                      ? 'Waiting for transmissions...' 
                      : 'Start monitoring to view packets'}
                  </p>
                  <div className="mt-6 bg-gray-800 rounded-lg p-4 max-w-2xl mx-auto text-left">
                    <p className="text-sm text-gray-400 mb-2">
                      To generate packets, run the sender script:
                    </p>
                    <code className="block bg-gray-900 p-3 rounded text-xs font-mono text-primary">
                      python scripts/run_sender.py --mode auto --sequence-length 10
                    </code>
                  </div>
                </div>
              ) : (
                packets.slice().reverse().map((packet) => (
                  <div
                    key={packet.filename}
                    className="bg-gray-800 border border-gray-700 rounded-lg p-4 hover:border-primary/50 transition-colors animate-fade-in"
                  >
                    <div className="flex items-start justify-between mb-2">
                      <div className="flex items-center space-x-3">
                        <span className="text-xs text-gray-500">
                          SEQ #{packet.sequence_number.toString().padStart(4, '0')}
                        </span>
                        <span className="font-mono text-sm text-primary">{packet.media_id}</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <Badge variant="info">CH-{packet.channel_id}</Badge>
                        <Badge variant={getModeColor(packet.mode_used)}>
                          {packet.mode_used}
                        </Badge>
                      </div>
                    </div>
                    <div className="grid grid-cols-3 gap-4 text-sm">
                      <div>
                        <span className="text-gray-400">Delay:</span>
                        <span className="ml-2 text-white font-mono">
                          {packet.delay_seconds.toFixed(2)}s
                        </span>
                      </div>
                      <div>
                        <span className="text-gray-400">Channel:</span>
                        <span className="ml-2 text-white">{packet.channel_id}</span>
                      </div>
                      <div>
                        <span className="text-gray-400">Time:</span>
                        <span className="ml-2 text-white font-mono">
                          {new Date(packet.timestamp * 1000).toLocaleTimeString()}
                        </span>
                      </div>
                    </div>
                  </div>
                ))
              )}
            </div>
          </Card>

          {/* Info Box */}
          <div className="mt-8 bg-gray-900 border border-gray-800 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-primary mb-3">How Wire View Works</h3>
            <div className="space-y-2 text-sm text-gray-300">
              <p>
                <strong className="text-white">Real-time Monitoring:</strong> This view watches the <code className="text-primary font-mono">shared_channel/</code> directory for new packet metadata files.
              </p>
              <p>
                <strong className="text-white">Dynamic Fallback:</strong> The system automatically tries RL → GAN → Static scheduling. The badge shows which mode is currently active.
              </p>
              <p>
                <strong className="text-white">Packet Metadata:</strong> Each JSON file contains media_id, channel, sequence number, delay, timestamp, and mode used.
              </p>
              <p>
                <strong className="text-white">Multi-Channel:</strong> Packets are distributed across multiple channels to mimic natural human behavior.
              </p>
            </div>
          </div>
        </div>
      </main>
    </>
  );
}
