import React, { useState, useEffect, useRef } from 'react';
import { 
  Activity, CheckCircle, Clock, AlertTriangle, 
  Monitor, Search, Bell, Ticket, Mail, Settings, 
  Shield, GitBranch, TrendingUp, Zap, 
  RefreshCw, ExternalLink, Eye, X, Terminal,
  Database, Wifi, Server, Lock, Container, HardDrive,
  Brain, MessageSquare, Network, Share2, Layers, Target,
  Play, Pause, BarChart3, Globe, Users, ChevronDown,
  ChevronUp, Code, FileText, Clock3
} from 'lucide-react';

function App() {
  const [dashboardStats, setDashboardStats] = useState({});
  const [agents, setAgents] = useState({});
  const [incidents, setIncidents] = useState([]);
  const [selectedIncident, setSelectedIncident] = useState(null);
  const [selectedAgent, setSelectedAgent] = useState(null);
  const [agentLogs, setAgentLogs] = useState(null);
  const [showAgentLogsModal, setShowAgentLogsModal] = useState(false);
  const [mcpContexts, setMcpContexts] = useState([]);
  const [a2aMessages, setA2aMessages] = useState([]);
  const [a2aCollaborations, setA2aCollaborations] = useState([]);
  const [showMcpModal, setShowMcpModal] = useState(false);
  const [showA2aModal, setShowA2aModal] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState(new Date());
  const [activeWorkflows, setActiveWorkflows] = useState(new Set());
  const [realTimeUpdates, setRealTimeUpdates] = useState([]);
  const [isConnected, setIsConnected] = useState(false);
  const [expandedLogTypes, setExpandedLogTypes] = useState(new Set());
  const websocketRef = useRef(null);

  useEffect(() => {
    fetchAllData();
    setupWebSocket();
    const interval = setInterval(fetchAllData, 3000);
    return () => {
      clearInterval(interval);
      if (websocketRef.current) {
        websocketRef.current.close();
      }
    };
  }, []);

  const setupWebSocket = () => {
    const wsUrl = `ws://${window.location.host}/ws/realtime`;
    const ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
      setIsConnected(true);
      console.log('WebSocket connected for real-time updates');
    };
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      handleRealTimeUpdate(data);
    };
    
    ws.onclose = () => {
      setIsConnected(false);
      console.log('WebSocket disconnected');
      setTimeout(setupWebSocket, 3000);
    };
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setIsConnected(false);
    };
    
    websocketRef.current = ws;
  };

  const handleRealTimeUpdate = (data) => {
    setRealTimeUpdates(prev => [data, ...prev.slice(0, 49)]);
    
    switch (data.type) {
      case 'mcp_update':
        fetchMcpContexts();
        break;
      case 'a2a_update':
        fetchA2aData();
        break;
      case 'workflow_update':
        fetchAllData();
        break;
      default:
        break;
    }
  };

  const fetchAllData = async () => {
    try {
      const [statsRes, agentsRes, incidentsRes] = await Promise.all([
        fetch('/api/dashboard/stats'),
        fetch('/api/agents'),
        fetch('/api/incidents?limit=10')
      ]);

      const [statsData, agentsData, incidentsData] = await Promise.all([
        statsRes.json(),
        agentsRes.json(),
        incidentsRes.json()
      ]);

      setDashboardStats(statsData);
      setAgents(agentsData.agents || {});
      setIncidents(incidentsData.incidents || []);
      setLastUpdate(new Date());
      setIsLoading(false);
      
      const activeIds = new Set(incidentsData.incidents
        .filter(i => i.workflow_status === 'in_progress')
        .map(i => i.id));
      setActiveWorkflows(activeIds);
      
    } catch (err) {
      console.error('Failed to fetch data:', err);
      setIsLoading(false);
    }
  };

  const fetchMcpContexts = async () => {
    try {
      const response = await fetch('/api/mcp/contexts');
      const data = await response.json();
      setMcpContexts(data.contexts || []);
    } catch (err) {
      console.error('Failed to fetch MCP contexts:', err);
    }
  };

  const fetchA2aData = async () => {
    try {
      const [messagesRes, collabRes] = await Promise.all([
        fetch('/api/a2a/messages/history?limit=20'),
        fetch('/api/a2a/collaborations')
      ]);
      
      const [messagesData, collabData] = await Promise.all([
        messagesRes.json(),
        collabRes.json()
      ]);
      
      setA2aMessages(messagesData.recent_messages || []);
      setA2aCollaborations(collabData.collaborations || []);
    } catch (err) {
      console.error('Failed to fetch A2A data:', err);
    }
  };

  const triggerTestIncident = async () => {
    try {
      const response = await fetch('/api/trigger-incident', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          title: '', 
          description: '',
          severity: 'high'
        })
      });
      const result = await response.json();
      
      const alertMessage = `🚀 oh-oh there is an Incident!\n\n` +
                          `Type: ${result.incident_type}\n` +
                          `Severity: ${result.severity}\n` +
                          `ID: ${result.incident_id}\n` +
                          `Business Impact: Available\n\n` +
                          `Title: ${result.title}\n\n` +
                          `✨ 7 AGENTS + MCP + A2A Features Active!\n` +
                          `🧠 Model Context Protocol: Shared intelligence\n` +
                          `🤝 Agent-to-Agent Protocol: Direct collaboration\n` +
                          `📊 Real-time updates via WebSocket\n` +
                          `📝 Detailed Console Logs: Click agents to view!`;
      
      alert(alertMessage);
      fetchAllData();
      
    } catch (err) {
      console.error('Failed to trigger incident:', err);
      alert('Failed to trigger incident. Please try again.');
    }
  };

  const viewIncidentDetails = async (incidentId) => {
    try {
      const response = await fetch(`/api/incidents/${incidentId}/status`);
      const incidentData = await response.json();
      setSelectedIncident(incidentData);
    } catch (err) {
      console.error('Failed to fetch incident details:', err);
    }
  };

  // FIXED: viewAgentLogs function now gracefully handles cases where logs are not found (404).
  const viewAgentLogs = async (agentId, incidentId = null) => {
    try {
      let targetIncidentId = incidentId;
      
      if (!targetIncidentId) {
        const sortedIncidents = [...incidents].sort((a, b) => 
          new Date(b.created_at) - new Date(a.created_at)
        );
        
        for (const incident of sortedIncidents) {
          try {
            const incidentResponse = await fetch(`/api/incidents/${incident.id}/status`);
            const incidentData = await incidentResponse.json();
            
            if (incidentData.executions && incidentData.executions[agentId]) {
              targetIncidentId = incident.id;
              break;
            }
          } catch (e) {
            console.log('Error checking incident:', incident.id, e);
            continue;
          }
        }
      }
      
      if (!targetIncidentId) {
        alert(`No execution found for ${agentId} agent. Please trigger an incident first to see detailed logs.`);
        return;
      }
      
      console.log(`Fetching logs for agent ${agentId} in incident ${targetIncidentId}`);
      const response = await fetch(`/api/incidents/${targetIncidentId}/agent/${agentId}/logs`);
      
      // Gracefully handle 404 Not Found. This stops the "unresponsive button" issue.
      if (response.status === 404) {
        console.warn(`No logs found (404) for agent ${agentId} in incident ${targetIncidentId}.`);
        const agentName = agents[agentId]?.name || agentId;
        setAgentLogs({
          agent_name: `${agentName}`,
          execution_id: 'N/A',
          status: 'Logs Not Found',
          duration_seconds: 0,
          progress: 0,
          log_summary: { total_log_entries: 0 },
          detailed_logs: [],
          mcp_enhancements: {},
          a2a_communications: {},
          business_context: {}
        });
        setSelectedAgent(agentId);
        setShowAgentLogsModal(true);
        setExpandedLogTypes(new Set());
        return; // Exit function after setting state to show the modal
      }
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const logsData = await response.json();
      
      if (logsData.error) {
        throw new Error(logsData.error);
      }
      
      setAgentLogs(logsData);
      setSelectedAgent(agentId);
      setShowAgentLogsModal(true);
      setExpandedLogTypes(new Set());
      
    } catch (err) {
      console.error('Failed to fetch agent logs:', err);
      alert(`Failed to fetch agent logs: ${err.message}`);
    }
  };

  const viewAgentLogsFromDashboard = async (agentId) => {
    await viewAgentLogs(agentId);
  };

  const getAgentIcon = (agentName) => {
    const icons = {
      monitoring: Monitor, rca: Search, pager: Bell,
      ticketing: Ticket, email: Mail, remediation: Settings, validation: Shield
    };
    return icons[agentName] || Activity;
  };

  const getIncidentTypeIcon = (incidentType) => {
    const icons = {
      database: Database, security: Lock, network: Wifi,
      infrastructure: Server, container: Container, storage: HardDrive,
      api: Activity, dns: Wifi, authentication: Lock,
      business_critical: Target, payment_critical: Lock,
      performance_critical: TrendingUp, trading_critical: BarChart3,
      business_anomaly: AlertTriangle, conversion_critical: Target,
      security_business: Shield
    };
    return icons[incidentType] || AlertTriangle;
  };

  const getIncidentTypeColor = (incidentType) => {
    const colors = {
      database: 'text-blue-400', security: 'text-red-400', network: 'text-green-400',
      infrastructure: 'text-purple-400', container: 'text-cyan-400', storage: 'text-yellow-400',
      api: 'text-pink-400', dns: 'text-indigo-400', authentication: 'text-orange-400',
      business_critical: 'text-red-500', payment_critical: 'text-yellow-500',
      performance_critical: 'text-blue-500', trading_critical: 'text-green-500',
      business_anomaly: 'text-purple-500', conversion_critical: 'text-pink-500',
      security_business: 'text-red-600'
    };
    return colors[incidentType] || 'text-gray-400';
  };

  const formatLogTimestamp = (timestamp) => {
    return new Date(timestamp).toLocaleTimeString();
  };

  const getLogTypeColor = (logType) => {
    const colors = {
      'SUCCESS': 'text-green-400',
      'ERROR': 'text-red-400',
      'BUSINESS_ANALYSIS': 'text-blue-400',
      'FINANCIAL_ANALYSIS': 'text-yellow-400',
      'TECHNICAL_ANALYSIS': 'text-purple-400',
      'MCP_ANALYSIS': 'text-cyan-400',
      'A2A_COLLABORATION': 'text-pink-400',
      'STAKEHOLDER_ANALYSIS': 'text-indigo-400',
      'ROOT_CAUSE_ANALYSIS': 'text-orange-400',
      'CLASSIFICATION': 'text-teal-400',
      'INFO': 'text-gray-400'
    };
    return colors[logType] || 'text-gray-400';
  };

  const getLogTypeIcon = (logType) => {
    const icons = {
      'SUCCESS': CheckCircle,
      'ERROR': AlertTriangle,
      'BUSINESS_ANALYSIS': Target,
      'FINANCIAL_ANALYSIS': TrendingUp,
      'TECHNICAL_ANALYSIS': Settings,
      'MCP_ANALYSIS': Brain,
      'A2A_COLLABORATION': MessageSquare,
      'STAKEHOLDER_ANALYSIS': Users,
      'ROOT_CAUSE_ANALYSIS': Search,
      'CLASSIFICATION': BarChart3,
      'INFO': Activity
    };
    return icons[logType] || Activity;
  };

  const toggleLogTypeExpansion = (logType) => {
    const newExpanded = new Set(expandedLogTypes);
    if (newExpanded.has(logType)) {
      newExpanded.delete(logType);
    } else {
      newExpanded.add(logType);
    }
    setExpandedLogTypes(newExpanded);
  };

  // Group logs by type for better organization
  const groupLogsByType = (logs) => {
    const grouped = {};
    logs.forEach(log => {
      const type = log.log_type || 'INFO';
      if (!grouped[type]) {
        grouped[type] = [];
      }
      grouped[type].push(log);
    });
    return grouped;
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-gray-900 flex items-center justify-center">
        <div className="text-center">
          <div className="flex items-center justify-center mb-4">
            <Brain className="w-8 h-8 text-purple-400 animate-pulse mr-2" />
            <MessageSquare className="w-8 h-8 text-blue-400 animate-bounce mr-2" />
            <Network className="w-8 h-8 text-green-400 animate-spin" />
          </div>
          <h2 className="text-2xl font-bold text-white mb-2">Loading Complete MCP + A2A Enhanced System</h2>
          <p className="text-gray-400">Initializing all 7 agents with Model Context Protocol and Agent-to-Agent Communication...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-gray-900">
      <style jsx global>{`
        .glass {
          background: rgba(255, 255, 255, 0.1);
          backdrop-filter: blur(10px);
          border: 1px solid rgba(255, 255, 255, 0.2);
        }
        .mcp-glow { box-shadow: 0 0 20px rgba(168, 85, 247, 0.3); }
        .a2a-glow { box-shadow: 0 0 20px rgba(59, 130, 246, 0.3); }
        .agent-glow { box-shadow: 0 0 15px rgba(34, 197, 94, 0.3); }
        .logs-glow { box-shadow: 0 0 25px rgba(255, 215, 0, 0.3); }
      `}</style>
      
      <header className="glass border-b border-purple-700/50">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="p-2 bg-gradient-to-br from-purple-500/20 to-blue-500/20 rounded-xl">
                <div className="flex space-x-1">
                  <Brain className="w-6 h-6 text-purple-400" />
                  <MessageSquare className="w-6 h-6 text-blue-400" />
                  <Terminal className="w-6 h-6 text-yellow-400" />
                </div>
              </div>
              <div>
                <h1 className="text-2xl font-bold text-white">OpsIntellect - MCP + A2A + Detailed Logs AI System</h1>
                <p className="text-sm text-gray-300">Multi-Agent • Model Context Protocol • Agent-to-Agent Communication • Enhanced Console Logging</p>
              </div>
              {activeWorkflows.size > 0 && (
                <div className="flex items-center space-x-2 ml-8 bg-gradient-to-r from-purple-500/20 to-blue-500/20 px-3 py-1 rounded-lg">
                  <Network className="w-4 h-4 text-purple-400 animate-spin" />
                  <span className="text-purple-400 font-medium">{activeWorkflows.size} Enhanced Workflows</span>
                </div>
              )}
            </div>
            <div className="text-right">
              <div className="flex items-center space-x-2">
                <div className={`w-2 h-2 rounded-full animate-pulse ${isConnected ? 'bg-green-400' : 'bg-red-400'}`}></div>
                <p className="text-sm font-medium text-white">
                  {isConnected ? 'Real-time Connected' : 'Reconnecting...'}
                </p>
              </div>
              <p className="text-xs text-gray-400">Last Updated: {lastUpdate.toLocaleTimeString()}</p>
            </div>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-6 py-8">
        {/* Enhanced Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-6 gap-4 mb-8">
          <div className="glass agent-glow rounded-xl p-4 hover:bg-green-500/10 transition-all">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-green-300">All 7 Agents</p>
                <p className="text-xl font-bold text-green-400">{Object.keys(agents).length}</p>
                <p className="text-xs text-green-500 mt-1">Enhanced & Logged</p>
              </div>
              <Users className="w-6 h-6 text-green-400" />
            </div>
          </div>

          <div className="glass mcp-glow rounded-xl p-4 hover:bg-purple-500/10 transition-all">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-purple-300">MCP Contexts</p>
                <p className="text-xl font-bold text-purple-400">{dashboardStats.enhanced_features?.mcp?.total_contexts || 0}</p>
                <p className="text-xs text-purple-500 mt-1">Shared Intelligence</p>
              </div>
              <Brain className="w-6 h-6 text-purple-400" />
            </div>
          </div>

          <div className="glass a2a-glow rounded-xl p-4 hover:bg-blue-500/10 transition-all">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-blue-300">A2A Messages</p>
                <p className="text-xl font-bold text-blue-400">{dashboardStats.enhanced_features?.a2a?.total_messages || 0}</p>
                <p className="text-xs text-blue-500 mt-1">Agent Collaboration</p>
              </div>
              <MessageSquare className="w-6 h-6 text-blue-400" />
            </div>
          </div>

          <div className="glass logs-glow rounded-xl p-4 hover:bg-yellow-500/10 transition-all">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-yellow-300">Console Logs</p>
                <p className="text-xl font-bold text-yellow-400">Active</p>
                <p className="text-xs text-yellow-500 mt-1">Click Agents!</p>
              </div>
              <Terminal className="w-6 h-6 text-yellow-400" />
            </div>
          </div>

          <div className="glass rounded-xl p-4 hover:bg-orange-500/10 transition-all">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-orange-300">Active Incidents</p>
                <p className="text-xl font-bold text-orange-400">{dashboardStats.incidents?.active || 0}</p>
                <p className="text-xs text-orange-500 mt-1">Business Focus</p>
              </div>
              <AlertTriangle className="w-6 h-6 text-orange-400" />
            </div>
          </div>

          <div className="glass rounded-xl p-4 hover:bg-pink-500/10 transition-all">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-pink-300">Success Rate</p>
                <p className="text-xl font-bold text-pink-400">
                  {Math.round(dashboardStats.system?.overall_success_rate || 95)}%
                </p>
                <p className="text-xs text-pink-500 mt-1">Enhanced</p>
              </div>
              <Target className="w-6 h-6 text-pink-400" />
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-3 gap-8">
          {/* ALL 7 AGENTS DASHBOARD - WITH ENHANCED CLICK FUNCTIONALITY */}
          <div className="xl:col-span-2">
            <div className="glass agent-glow rounded-xl p-6">
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-xl font-semibold text-white">Multi AI Agents - Click for Console Logs</h3>
                <div className="flex items-center space-x-4">
                  <div className="flex items-center space-x-1">
                    <Terminal className="w-4 h-4 text-yellow-400" />
                    <span className="text-sm text-yellow-400">Logs</span>
                  </div>
                  <div className="flex items-center space-x-1">
                    <Brain className="w-4 h-4 text-purple-400" />
                    <span className="text-sm text-purple-400">MCP</span>
                  </div>
                  <div className="flex items-center space-x-1">
                    <MessageSquare className="w-4 h-4 text-blue-400" />
                    <span className="text-sm text-blue-400">A2A</span>
                  </div>
                  <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                  <span className="text-sm text-green-400">All Ready</span>
                </div>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {Object.entries(agents).map(([agentId, agent]) => {
                  const IconComponent = getAgentIcon(agentId);
                  return (
                    <div 
                      key={agentId} 
                      className="bg-gradient-to-br from-gray-800/50 to-purple-900/20 rounded-lg p-4 border border-purple-600/30 hover:border-yellow-500/70 transition-all cursor-pointer transform hover:scale-[1.02] hover:shadow-lg hover:shadow-yellow-500/20"
                      onClick={() => viewAgentLogsFromDashboard(agentId)}
                    >
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center space-x-3">
                          <div className="p-2 bg-gradient-to-br from-purple-500/20 to-blue-500/20 rounded-lg">
                            <IconComponent className="w-5 h-5 text-purple-400" />
                          </div>
                          <div>
                            <span className="font-medium text-white capitalize">{agentId}</span>
                            <p className="text-xs text-purple-300">Enhanced Agent</p>
                          </div>
                        </div>
                        <div className="flex items-center space-x-1">
                          <Terminal className="w-3 h-3 text-yellow-400 animate-pulse" />
                          <span className="text-xs text-green-400 font-medium">Ready</span>
                        </div>
                      </div>
                      
                      <p className="text-sm text-gray-300 mb-3 line-clamp-2">{agent.description}</p>
                      
                      <div className="grid grid-cols-2 gap-2 text-xs mb-3">
                        <div>
                          <span className="text-gray-400">Executions:</span>
                          <span className="text-purple-400 font-medium ml-1">{agent.total_executions}</span>
                        </div>
                        <div>
                          <span className="text-gray-400">Success:</span>
                          <span className="text-green-400 font-medium ml-1">{agent.success_rate?.toFixed(1)}%</span>
                        </div>
                        <div>
                          <span className="text-gray-400">Avg Time:</span>
                          <span className="text-blue-400 font-medium ml-1">{agent.average_duration?.toFixed(1)}s</span>
                        </div>
                        <div>
                          <span className="text-gray-400">Logs:</span>
                          <span className="text-yellow-400 font-medium ml-1">{agent.enhanced_features?.detailed_logging?.total_logs || 0}</span>
                        </div>
                      </div>
                      
                      <div className="flex items-center justify-between">
                        <div className="flex space-x-1">
                          {agent.enhanced_features?.detailed_logging?.total_logs > 0 && (
                            <Terminal className="w-3 h-3 text-yellow-400" title="Detailed Logs Available" />
                          )}
                          {agent.enhanced_features?.mcp_enhanced_executions > 0 && (
                            <Brain className="w-3 h-3 text-purple-400" title="MCP Enhanced" />
                          )}
                          {agent.enhanced_features?.a2a_messages_total > 0 && (
                            <MessageSquare className="w-3 h-3 text-blue-400" title="A2A Active" />
                          )}
                          <Layers className="w-3 h-3 text-green-400" title="Enhanced" />
                        </div>
                        <div className="flex items-center space-x-1">
                          <Terminal className="w-3 h-3 text-yellow-400" />
                          <span className="text-xs text-yellow-400">Click for Console Logs</span>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Enhanced Controls & Real-time Feed */}
          <div className="space-y-6">
            <div className="glass rounded-xl p-6">
              <h3 className="text-xl font-semibold text-white mb-4">Enhanced Controls</h3>
              <div className="space-y-3">
                <button
                  onClick={triggerTestIncident}
                  className="w-full bg-gradient-to-r from-purple-500 via-pink-500 to-red-500 hover:from-purple-600 hover:via-pink-600 hover:to-red-600 text-white px-4 py-3 rounded-lg font-medium transition-all duration-300 flex items-center justify-center space-x-2 shadow-lg transform hover:scale-105"
                >
                  <div className="flex space-x-1">
                    <Brain className="w-4 h-4" />
                    <MessageSquare className="w-4 h-4" />
                    <Terminal className="w-4 h-4" />
                    <Users className="w-4 h-4" />
                  </div>
                  <span>Check For Incident</span>
                </button>
                <p className="text-xs text-gray-400 text-center">
                  Enhanced Agents + MCP + A2A + Console Logs
                </p>
                
                <div className="grid grid-cols-2 gap-2">
                  <button 
                    onClick={() => {setShowMcpModal(true); fetchMcpContexts();}}
                    className="bg-gradient-to-r from-purple-500/20 to-purple-600/20 border border-purple-500/50 text-purple-300 px-3 py-2 rounded-lg text-sm font-medium hover:bg-purple-500/30 transition-all flex items-center justify-center space-x-1"
                  >
                    <Brain className="w-3 h-3" />
                    <span>MCP Status</span>
                  </button>
                  
                  <button 
                    onClick={() => {setShowA2aModal(true); fetchA2aData();}}
                    className="bg-gradient-to-r from-blue-500/20 to-blue-600/20 border border-blue-500/50 text-blue-300 px-3 py-2 rounded-lg text-sm font-medium hover:bg-blue-500/30 transition-all flex items-center justify-center space-x-1"
                  >
                    <MessageSquare className="w-3 h-3" />
                    <span>A2A Network</span>
                  </button>
                </div>
                
                <button 
                  onClick={fetchAllData}
                  className="w-full bg-gradient-to-r from-green-500/20 to-emerald-500/20 border border-green-500/50 text-green-300 px-4 py-2 rounded-lg font-medium hover:bg-green-500/30 transition-all flex items-center justify-center space-x-2"
                >
                  <RefreshCw className="w-4 h-4" />
                  <span>Refresh All Data</span>
                </button>
              </div>
            </div>

            {/* Real-time Updates Feed */}
            <div className="glass rounded-xl p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xl font-semibold text-white">Real-time Updates</h3>
                <div className="flex items-center space-x-2">
                  <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-400 animate-pulse' : 'bg-red-400'}`}></div>
                  <span className="text-xs text-gray-400">{isConnected ? 'Connected' : 'Disconnected'}</span>
                </div>
              </div>
              
              <div className="space-y-2 max-h-64 overflow-y-auto">
                {realTimeUpdates.length === 0 ? (
                  <p className="text-sm text-gray-400 text-center py-4">Waiting for real-time updates...</p>
                ) : (
                  realTimeUpdates.slice(0, 10).map((update, index) => (
                    <div key={index} className="bg-gray-800/30 rounded p-2 text-xs">
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-purple-400 font-medium">{update.type}</span>
                        <span className="text-gray-500">{new Date(update.timestamp).toLocaleTimeString()}</span>
                      </div>
                      <p className="text-gray-300">{update.message || 'System update'}</p>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>

          {/* Recent Incidents */}
          <div className="xl:col-span-3">
            <div className="glass rounded-xl p-6">
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-xl font-semibold text-white">Recent Incidents History</h3>
                <div className="flex items-center space-x-2">
                  <Target className="w-4 h-4 text-orange-400" />
                  <span className="text-sm text-orange-400">Business Focus</span>
                </div>
              </div>
              
              <div className="space-y-3 max-h-96 overflow-y-auto">
                {incidents.length === 0 ? (
                  <div className="text-center py-8">
                    <AlertTriangle className="w-12 h-12 text-gray-500 mx-auto mb-4" />
                    <p className="text-gray-400 mb-4">No recent incidents</p>
                    <button
                      onClick={triggerTestIncident}
                      className="bg-purple-500 hover:bg-purple-600 text-white px-4 py-2 rounded-lg transition-all"
                    >
                      Trigger Test Incident
                    </button>
                  </div>
                ) : (
                  incidents.map((incident) => {
                    const IconComponent = getIncidentTypeIcon(incident.incident_type);
                    const colorClass = getIncidentTypeColor(incident.incident_type);
                    
                    return (
                      <div 
                        key={incident.id} 
                        className="bg-gradient-to-r from-gray-800/50 to-purple-900/20 rounded-lg p-4 border border-gray-700/50 hover:border-purple-500/50 transition-all cursor-pointer"
                        onClick={() => viewIncidentDetails(incident.id)}
                      >
                        <div className="flex items-start justify-between mb-3">
                          <div className="flex items-center space-x-3">
                            <div className={`p-2 bg-gray-800/50 rounded-lg`}>
                              <IconComponent className={`w-5 h-5 ${colorClass}`} />
                            </div>
                            <div className="flex-1">
                              <h4 className="font-medium text-white text-sm mb-1">{incident.title}</h4>
                              <p className="text-xs text-gray-400 line-clamp-2">{incident.description}</p>
                            </div>
                          </div>
                          <div className="text-right">
                            <span className={`text-xs px-2 py-1 rounded ${
                              incident.severity === 'critical' ? 'bg-red-500/20 text-red-400' :
                              incident.severity === 'high' ? 'bg-orange-500/20 text-orange-400' :
                              'bg-yellow-500/20 text-yellow-400'
                            }`}>
                              {incident.severity}
                            </span>
                          </div>
                        </div>
                        
                        <div className="flex items-center justify-between text-xs">
                          <div className="flex items-center space-x-3">
                            <span className="text-gray-400">Type:</span>
                            <span className={`${colorClass} font-medium`}>{incident.incident_type}</span>
                          </div>
                          <div className="flex items-center space-x-2">
                            <span className="text-gray-400">{new Date(incident.created_at).toLocaleTimeString()}</span>
                            {incident.detailed_logs_available > 0 && (
                              <div className="flex items-center space-x-1">
                                <Terminal className="w-3 h-3 text-yellow-400" />
                                <span className="text-yellow-400">{incident.detailed_logs_available} logs</span>
                              </div>
                            )}
                          </div>
                        </div>
                        
                        {incident.business_impact && (
                          <div className="mt-2 pt-2 border-t border-gray-700/50">
                            <p className="text-xs text-blue-300">💼 {incident.business_impact}</p>
                          </div>
                        )}
                      </div>
                    );
                  })
                )}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* DETAILED AGENT LOGS MODAL - COMPLETE IMPLEMENTATION */}
      {showAgentLogsModal && agentLogs && (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4 z-50">
          <div className="bg-gradient-to-br from-gray-900/95 to-purple-900/95 rounded-xl border border-purple-500/50 w-full max-w-6xl max-h-[90vh] flex flex-col logs-glow">
            {/* Modal Header */}
            <div className="flex items-center justify-between p-6 border-b border-purple-500/30">
              <div className="flex items-center space-x-4">
                <div className="p-2 bg-gradient-to-br from-yellow-500/20 to-orange-500/20 rounded-lg">
                  <Terminal className="w-6 h-6 text-yellow-400" />
                </div>
                <div>
                  <h2 className="text-xl font-bold text-white">
                    {agentLogs.agent_name} - Console Logs
                  </h2>
                  <p className="text-sm text-gray-300">
                    Execution: {agentLogs.execution_id} | Status: {agentLogs.status}
                  </p>
                </div>
              </div>
              <button
                onClick={() => setShowAgentLogsModal(false)}
                className="p-2 hover:bg-gray-800/50 rounded-lg transition-all"
              >
                <X className="w-5 h-5 text-gray-400" />
              </button>
            </div>

            {/* Execution Summary */}
            <div className="p-6 border-b border-gray-700/50">
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
                <div className="bg-gray-800/30 rounded-lg p-3">
                  <div className="text-xs text-gray-400 mb-1">Duration</div>
                  <div className="text-lg font-bold text-blue-400">
                    {agentLogs.duration_seconds?.toFixed(1)}s
                  </div>
                </div>
                <div className="bg-gray-800/30 rounded-lg p-3">
                  <div className="text-xs text-gray-400 mb-1">Progress</div>
                  <div className="text-lg font-bold text-green-400">
                    {agentLogs.progress}%
                  </div>
                </div>
                <div className="bg-gray-800/30 rounded-lg p-3">
                  <div className="text-xs text-gray-400 mb-1">Total Logs</div>
                  <div className="text-lg font-bold text-yellow-400">
                    {agentLogs.log_summary?.total_log_entries || 0}
                  </div>
                </div>
                <div className="bg-gray-800/30 rounded-lg p-3">
                  <div className="text-xs text-gray-400 mb-1">Business Context</div>
                  <div className="text-lg font-bold text-purple-400">
                    {agentLogs.log_summary?.business_focused_logs || 0}
                  </div>
                </div>
              </div>

              {/* Enhanced Features */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-3">
                  <div className="flex items-center space-x-2 mb-2">
                    <Brain className="w-4 h-4 text-purple-400" />
                    <span className="text-sm font-medium text-purple-300">MCP Enhanced</span>
                  </div>
                  <div className="text-xs text-gray-300">
                    Context ID: {agentLogs.mcp_enhancements?.context_id?.slice(-8) || 'N/A'}
                  </div>
                  <div className="text-xs text-purple-400 mt-1">
                    {agentLogs.mcp_enhancements?.mcp_enhanced ? 'Active' : 'Inactive'}
                  </div>
                </div>

                <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-3">
                  <div className="flex items-center space-x-2 mb-2">
                    <MessageSquare className="w-4 h-4 text-blue-400" />
                    <span className="text-sm font-medium text-blue-300">A2A Communication</span>
                  </div>
                  <div className="text-xs text-gray-300">
                    Sent: {agentLogs.a2a_communications?.messages_sent || 0} | 
                    Received: {agentLogs.a2a_communications?.messages_received || 0}
                  </div>
                  <div className="text-xs text-blue-400 mt-1">
                    Collaborations: {agentLogs.a2a_communications?.collaboration_sessions?.length || 0}
                  </div>
                </div>

                <div className="bg-orange-500/10 border border-orange-500/30 rounded-lg p-3">
                  <div className="flex items-center space-x-2 mb-2">
                    <Target className="w-4 h-4 text-orange-400" />
                    <span className="text-sm font-medium text-orange-300">Business Context</span>
                  </div>
                  <div className="text-xs text-gray-300">
                    Type: {agentLogs.business_context?.incident_type || 'N/A'}
                  </div>
                  <div className="text-xs text-orange-400 mt-1">
                    Severity: {agentLogs.business_context?.severity || 'N/A'}
                  </div>
                </div>
              </div>
            </div>

            {/* Detailed Logs */}
            <div className="flex-1 overflow-hidden">
              <div className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold text-white">Detailed Console Logs</h3>
                  <div className="flex items-center space-x-2">
                    <span className="text-sm text-gray-400">
                      {agentLogs.log_summary?.log_types?.length || 0} log types
                    </span>
                  </div>
                </div>

                <div className="bg-black/40 rounded-lg border border-gray-700/50 max-h-96 overflow-y-auto">
                  {agentLogs.detailed_logs && agentLogs.detailed_logs.length > 0 ? (
                    <div className="p-4 space-y-3">
                      {/* Group logs by type */}
                      {Object.entries(groupLogsByType(agentLogs.detailed_logs)).map(([logType, logs]) => {
                        const LogIcon = getLogTypeIcon(logType);
                        const colorClass = getLogTypeColor(logType);
                        const isExpanded = expandedLogTypes.has(logType);
                        
                        return (
                          <div key={logType} className="border border-gray-700/30 rounded-lg">
                            <button
                              onClick={() => toggleLogTypeExpansion(logType)}
                              className="w-full flex items-center justify-between p-3 hover:bg-gray-800/30 transition-all"
                            >
                              <div className="flex items-center space-x-3">
                                <LogIcon className={`w-4 h-4 ${colorClass}`} />
                                <span className={`font-medium ${colorClass}`}>{logType}</span>
                                <span className="text-xs text-gray-500">({logs.length} entries)</span>
                              </div>
                              {isExpanded ? (
                                <ChevronUp className="w-4 h-4 text-gray-400" />
                              ) : (
                                <ChevronDown className="w-4 h-4 text-gray-400" />
                              )}
                            </button>
                            
                            {isExpanded && (
                              <div className="border-t border-gray-700/30 p-3 space-y-2">
                                {logs.map((log, index) => (
                                  <div key={index} className="bg-gray-900/50 rounded p-3 font-mono text-sm">
                                    <div className="flex items-center justify-between mb-2">
                                      <span className="text-gray-400 text-xs">
                                        {formatLogTimestamp(log.timestamp)}
                                      </span>
                                      {log.business_context && (
                                        <div className="flex items-center space-x-1">
                                          <Target className="w-3 h-3 text-orange-400" />
                                          <span className="text-xs text-orange-400">Business</span>
                                        </div>
                                      )}
                                    </div>
                                    <div className="text-gray-100 mb-2">{log.message}</div>
                                    {log.additional_data && Object.keys(log.additional_data).length > 0 && (
                                      <div className="bg-gray-800/50 rounded p-2 mt-2">
                                        <div className="text-xs text-gray-400 mb-1">Additional Data:</div>
                                        <pre className="text-xs text-gray-300 whitespace-pre-wrap">
                                          {JSON.stringify(log.additional_data, null, 2)}
                                        </pre>
                                      </div>
                                    )}
                                  </div>
                                ))}
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  ) : (
                    <div className="p-8 text-center">
                      <Terminal className="w-12 h-12 text-gray-500 mx-auto mb-4" />
                      <p className="text-gray-400 mb-2">No detailed logs available for this execution.</p>
                      <p className="text-xs text-gray-500">Logs will appear here after an agent runs and generates output.</p>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Modal Footer */}
            <div className="p-6 border-t border-gray-700/50">
              <div className="flex items-center justify-between">
                <div className="text-sm text-gray-400">
                  Last execution: {agentLogs.completed_at ? new Date(agentLogs.completed_at).toLocaleString() : 'In progress'}
                </div>
                <button
                  onClick={() => setShowAgentLogsModal(false)}
                  className="bg-purple-500 hover:bg-purple-600 text-white px-4 py-2 rounded-lg transition-all"
                >
                  Close Logs
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* MCP Modal */}
      {showMcpModal && (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4 z-50">
          <div className="bg-gradient-to-br from-gray-900/95 to-purple-900/95 rounded-xl border border-purple-500/50 w-full max-w-4xl max-h-[80vh] flex flex-col mcp-glow">
            <div className="flex items-center justify-between p-6 border-b border-purple-500/30">
              <div className="flex items-center space-x-4">
                <Brain className="w-6 h-6 text-purple-400" />
                <h2 className="text-xl font-bold text-white">Model Context Protocol Status</h2>
              </div>
              <button
                onClick={() => setShowMcpModal(false)}
                className="p-2 hover:bg-gray-800/50 rounded-lg transition-all"
              >
                <X className="w-5 h-5 text-gray-400" />
              </button>
            </div>
            
            <div className="flex-1 overflow-y-auto p-6">
              <div className="space-y-4">
                {mcpContexts.length === 0 ? (
                  <div className="text-center py-8">
                    <Brain className="w-12 h-12 text-gray-500 mx-auto mb-4" />
                    <p className="text-gray-400">No MCP contexts available</p>
                  </div>
                ) : (
                  mcpContexts.map((context) => (
                    <div key={context.context_id} className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
                      <div className="flex items-center justify-between mb-3">
                        <h3 className="font-medium text-purple-300">Context: {context.context_id.slice(-8)}</h3>
                        <span className="text-xs text-gray-400">{new Date(context.created_at).toLocaleString()}</span>
                      </div>
                      <div className="grid grid-cols-2 gap-4 text-sm">
                        <div>
                          <span className="text-gray-400">Incident:</span>
                          <span className="text-white ml-2">{context.incident_id}</span>
                        </div>
                        <div>
                          <span className="text-gray-400">Agents:</span>
                          <span className="text-purple-400 ml-2">{context.agent_count}</span>
                        </div>
                        <div>
                          <span className="text-gray-400">Type:</span>
                          <span className="text-white ml-2">{context.context_type}</span>
                        </div>
                        <div>
                          <span className="text-gray-400">Confidence:</span>
                          <span className="text-green-400 ml-2">{(context.confidence_avg * 100).toFixed(1)}%</span>
                        </div>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* A2A Modal */}
      {showA2aModal && (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4 z-50">
          <div className="bg-gradient-to-br from-gray-900/95 to-blue-900/95 rounded-xl border border-blue-500/50 w-full max-w-6xl max-h-[80vh] flex flex-col a2a-glow">
            <div className="flex items-center justify-between p-6 border-b border-blue-500/30">
              <div className="flex items-center space-x-4">
                <MessageSquare className="w-6 h-6 text-blue-400" />
                <h2 className="text-xl font-bold text-white">Agent-to-Agent Network</h2>
              </div>
              <button
                onClick={() => setShowA2aModal(false)}
                className="p-2 hover:bg-gray-800/50 rounded-lg transition-all"
              >
                <X className="w-5 h-5 text-gray-400" />
              </button>
            </div>
            
            <div className="flex-1 overflow-y-auto p-6">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Recent Messages */}
                <div>
                  <h3 className="text-lg font-semibold text-white mb-4">Recent Messages</h3>
                  <div className="space-y-3 max-h-64 overflow-y-auto">
                    {a2aMessages.length === 0 ? (
                      <p className="text-gray-400 text-center py-4">No messages yet</p>
                    ) : (
                      a2aMessages.map((message) => (
                        <div key={message.message_id} className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-3">
                          <div className="flex items-center justify-between mb-2">
                            <span className="text-blue-300 font-medium text-sm">
                              {message.sender} → {message.receiver}
                            </span>
                            <span className="text-xs text-gray-400">
                              {new Date(message.timestamp).toLocaleTimeString()}
                            </span>
                          </div>
                          <div className="text-sm">
                            <span className="text-gray-400">Type:</span>
                            <span className="text-white ml-2">{message.type}</span>
                          </div>
                          <div className="text-sm">
                            <span className="text-gray-400">Priority:</span>
                            <span className={`ml-2 ${message.priority === 'high' ? 'text-red-400' : 
                              message.priority === 'critical' ? 'text-red-500' : 'text-yellow-400'}`}>
                              {message.priority}
                            </span>
                          </div>
                        </div>
                      ))
                    )}
                  </div>
                </div>

                {/* Active Collaborations */}
                <div>
                  <h3 className="text-lg font-semibold text-white mb-4">Active Collaborations</h3>
                  <div className="space-y-3 max-h-64 overflow-y-auto">
                    {a2aCollaborations.length === 0 ? (
                      <p className="text-gray-400 text-center py-4">No active collaborations</p>
                    ) : (
                      a2aCollaborations.map((collab) => (
                        <div key={collab.collaboration_id} className="bg-green-500/10 border border-green-500/30 rounded-lg p-3">
                          <div className="flex items-center justify-between mb-2">
                            <span className="text-green-300 font-medium text-sm">
                              {collab.collaboration_id.slice(-8)}
                            </span>
                            <span className="text-xs text-gray-400">
                              {new Date(collab.created_at).toLocaleTimeString()}
                            </span>
                          </div>
                          <div className="text-sm mb-1">
                            <span className="text-gray-400">Task:</span>
                            <span className="text-white ml-2">{collab.task}</span>
                          </div>
                          <div className="text-sm">
                            <span className="text-gray-400">Participants:</span>
                            <span className="text-green-400 ml-2">{collab.participants.join(', ')}</span>
                          </div>
                        </div>
                      ))
                    )}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Incident Details Modal */}
      {selectedIncident && (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4 z-50">
          <div className="bg-gradient-to-br from-gray-900/95 to-purple-900/95 rounded-xl border border-purple-500/50 w-full max-w-6xl max-h-[90vh] flex flex-col">
            <div className="flex items-center justify-between p-6 border-b border-purple-500/30">
              <div className="flex items-center space-x-4">
                <div className="p-2 bg-gradient-to-br from-orange-500/20 to-red-500/20 rounded-lg">
                  <AlertTriangle className="w-6 h-6 text-orange-400" />
                </div>
                <div>
                  <h2 className="text-xl font-bold text-white">{selectedIncident.title}</h2>
                  <p className="text-sm text-gray-300">ID: {selectedIncident.incident_id}</p>
                </div>
              </div>
              <button
                onClick={() => setSelectedIncident(null)}
                className="p-2 hover:bg-gray-800/50 rounded-lg transition-all"
              >
                <X className="w-5 h-5 text-gray-400" />
              </button>
            </div>
            
            <div className="flex-1 overflow-y-auto p-6">
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Incident Details */}
                <div className="lg:col-span-2 space-y-6">
                  <div>
                    <h3 className="text-lg font-semibold text-white mb-3">Incident Overview</h3>
                    <div className="bg-gray-800/30 rounded-lg p-4">
                      <p className="text-gray-300 mb-4">{selectedIncident.description}</p>
                      
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <span className="text-gray-400">Severity:</span>
                          <span className={`ml-2 px-2 py-1 rounded text-xs ${
                            selectedIncident.severity === 'critical' ? 'bg-red-500/20 text-red-400' :
                            selectedIncident.severity === 'high' ? 'bg-orange-500/20 text-orange-400' :
                            'bg-yellow-500/20 text-yellow-400'
                          }`}>
                            {selectedIncident.severity}
                          </span>
                        </div>
                        <div>
                          <span className="text-gray-400">Status:</span>
                          <span className="text-white ml-2">{selectedIncident.status}</span>
                        </div>
                        <div>
                          <span className="text-gray-400">Type:</span>
                          <span className="text-purple-400 ml-2">{selectedIncident.incident_type}</span>
                        </div>
                        <div>
                          <span className="text-gray-400">Workflow:</span>
                          <span className="text-blue-400 ml-2">{selectedIncident.workflow_status}</span>
                        </div>
                      </div>
                      
                      {selectedIncident.business_impact && (
                        <div className="mt-4 p-3 bg-blue-500/10 border border-blue-500/30 rounded">
                          <div className="text-sm text-blue-300 font-medium mb-1">Business Impact:</div>
                          <div className="text-sm text-gray-300">{selectedIncident.business_impact}</div>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Agent Executions */}
                  <div>
                    <h3 className="text-lg font-semibold text-white mb-3">Agent Executions</h3>
                    <div className="space-y-3">
                      {Object.entries(selectedIncident.executions || {}).map(([agentId, execution]) => {
                        const IconComponent = getAgentIcon(agentId);
                        return (
                          <div key={agentId} className="bg-gray-800/30 rounded-lg p-4">
                            <div className="flex items-center justify-between mb-3">
                              <div className="flex items-center space-x-3">
                                <IconComponent className="w-5 h-5 text-purple-400" />
                                <span className="font-medium text-white capitalize">{agentId}</span>
                              </div>
                              <div className="flex items-center space-x-2">
                                <span className={`text-xs px-2 py-1 rounded ${
                                  execution.status === 'success' ? 'bg-green-500/20 text-green-400' :
                                  execution.status === 'error' ? 'bg-red-500/20 text-red-400' :
                                  execution.status === 'running' ? 'bg-blue-500/20 text-blue-400' :
                                  'bg-gray-500/20 text-gray-400'
                                }`}>
                                  {execution.status}
                                </span>
                                {execution.detailed_logging?.logs_available && (
                                  <button
                                    onClick={() => viewAgentLogs(agentId, selectedIncident.incident_id)}
                                    className="bg-yellow-500/20 hover:bg-yellow-500/30 text-yellow-400 px-2 py-1 rounded text-xs transition-all flex items-center space-x-1"
                                  >
                                    <Terminal className="w-3 h-3" />
                                    <span>View Logs</span>
                                  </button>
                                )}
                              </div>
                            </div>
                            
                            <div className="grid grid-cols-3 gap-4 text-sm">
                              <div>
                                <span className="text-gray-400">Duration:</span>
                                <span className="text-white ml-2">{execution.duration?.toFixed(1)}s</span>
                              </div>
                              <div>
                                <span className="text-gray-400">Progress:</span>
                                <span className="text-white ml-2">{execution.progress}%</span>
                              </div>
                              <div>
                                <span className="text-gray-400">Logs:</span>
                                <span className="text-yellow-400 ml-2">{execution.detailed_logging?.total_log_entries || 0}</span>
                              </div>
                            </div>
                            
                            {execution.mcp_enhanced && (
                              <div className="mt-2 flex items-center space-x-2">
                                <Brain className="w-3 h-3 text-purple-400" />
                                <span className="text-xs text-purple-400">MCP Enhanced</span>
                                {execution.a2a_messages?.sent > 0 && (
                                  <>
                                    <MessageSquare className="w-3 h-3 text-blue-400 ml-2" />
                                    <span className="text-xs text-blue-400">A2A: {execution.a2a_messages.sent + execution.a2a_messages.received} msgs</span>
                                  </>
                                )}
                              </div>
                            )}
                            
                            {execution.error && (
                              <div className="mt-2 p-2 bg-red-500/10 border border-red-500/30 rounded">
                                <div className="text-xs text-red-400">{execution.error}</div>
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>

                {/* Sidebar */}
                <div className="space-y-6">
                  {/* Progress */}
                  <div>
                    <h3 className="text-lg font-semibold text-white mb-3">Workflow Progress</h3>
                    <div className="bg-gray-800/30 rounded-lg p-4">
                      <div className="space-y-3">
                        <div>
                          <span className="text-gray-400">Completed Agents:</span>
                          <span className="text-green-400 ml-2">{selectedIncident.completed_agents?.length || 0}/7</span>
                        </div>
                        <div>
                          <span className="text-gray-400">Failed Agents:</span>
                          <span className="text-red-400 ml-2">{selectedIncident.failed_agents?.length || 0}</span>
                        </div>
                        <div>
                          <span className="text-gray-400">Current Agent:</span>
                          <span className="text-blue-400 ml-2">{selectedIncident.current_agent || 'None'}</span>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Enhanced Features */}
                  {selectedIncident.enhanced_features && (
                    <div>
                      <h3 className="text-lg font-semibold text-white mb-3">Enhanced Features</h3>
                      <div className="space-y-3">
                        {selectedIncident.enhanced_features.mcp_context && (
                          <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-3">
                            <div className="flex items-center space-x-2 mb-2">
                              <Brain className="w-4 h-4 text-purple-400" />
                              <span className="text-sm font-medium text-purple-300">MCP Context</span>
                            </div>
                            <div className="text-xs text-gray-300">
                              Version: {selectedIncident.enhanced_features.mcp_context.context_version}
                            </div>
                            <div className="text-xs text-purple-400">
                              Confidence: {(selectedIncident.enhanced_features.mcp_context.avg_confidence * 100).toFixed(1)}%
                            </div>
                          </div>
                        )}

                        {selectedIncident.enhanced_features.a2a_protocol && (
                          <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-3">
                            <div className="flex items-center space-x-2 mb-2">
                              <MessageSquare className="w-4 h-4 text-blue-400" />
                              <span className="text-sm font-medium text-blue-300">A2A Protocol</span>
                            </div>
                            <div className="text-xs text-gray-300">
                              Total Messages: {selectedIncident.enhanced_features.a2a_protocol.total_messages_sent + selectedIncident.enhanced_features.a2a_protocol.total_messages_received}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Resolution */}
                  {selectedIncident.resolution && (
                    <div>
                      <h3 className="text-lg font-semibold text-white mb-3">Resolution</h3>
                      <div className="bg-gray-800/30 rounded-lg p-4">
                        <p className="text-sm text-gray-300">{selectedIncident.resolution}</p>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;