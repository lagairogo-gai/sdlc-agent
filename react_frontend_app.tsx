import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { 
  Play, Pause, Square, Settings, Monitor, Code, TestTube, 
  Rocket, Eye, CheckCircle, AlertCircle, Clock, Activity,
  ChevronRight, ChevronDown, Layers, Zap, Database, Shield,
  GitBranch, Package, BarChart3, MessageSquare, Bell
} from 'lucide-react';

// Theme definitions
const themes = {
  dark: {
    primary: '#1a1a1a',
    secondary: '#2d2d2d',
    accent: '#3b82f6',
    success: '#10b981',
    warning: '#f59e0b',
    error: '#ef4444',
    text: '#ffffff',
    textSecondary: '#a1a1aa',
    border: '#404040',
    background: '#0f0f0f',
    cardBg: '#1a1a1a',
    gradientFrom: '#1a1a1a',
    gradientTo: '#2d2d2d'
  },
  light: {
    primary: '#ffffff',
    secondary: '#f8fafc',
    accent: '#3b82f6',
    success: '#10b981',
    warning: '#f59e0b',
    error: '#ef4444',
    text: '#1f1f1f',
    textSecondary: '#6b7280',
    border: '#e5e7eb',
    background: '#f9fafb',
    cardBg: '#ffffff',
    gradientFrom: '#ffffff',
    gradientTo: '#f8fafc'
  },
  neon: {
    primary: '#0a0a0a',
    secondary: '#1a1a2e',
    accent: '#00ff88',
    success: '#00ff88',
    warning: '#ffaa00',
    error: '#ff0055',
    text: '#ffffff',
    textSecondary: '#88ddff',
    border: '#00ff88',
    background: '#000011',
    cardBg: '#1a1a2e',
    gradientFrom: '#0a0a0a',
    gradientTo: '#1a1a2e'
  },
  ocean: {
    primary: '#0f172a',
    secondary: '#1e293b',
    accent: '#0ea5e9',
    success: '#06b6d4',
    warning: '#f97316',
    error: '#e11d48',
    text: '#f1f5f9',
    textSecondary: '#94a3b8',
    border: '#334155',
    background: '#020617',
    cardBg: '#0f172a',
    gradientFrom: '#0f172a',
    gradientTo: '#1e293b'
  }
};

// Mock data for agents and workflow
const mockAgents = [
  {
    id: 'requirements_agent',
    name: 'Requirements Agent',
    type: 'requirements',
    status: 'running',
    progress: 75,
    icon: <MessageSquare className="w-5 h-5" />,
    description: 'Gathering and analyzing project requirements',
    tools: ['Jira', 'Confluence', 'Stakeholder Interviews'],
    currentTask: 'Conducting stakeholder interviews',
    logs: [
      { timestamp: '2025-01-23T10:30:00Z', level: 'info', message: 'Starting requirements gathering process' },
      { timestamp: '2025-01-23T10:31:00Z', level: 'info', message: 'Connected to Jira successfully' },
      { timestamp: '2025-01-23T10:32:00Z', level: 'success', message: 'Stakeholder interview scheduled' },
      { timestamp: '2025-01-23T10:33:00Z', level: 'info', message: 'Analyzing existing documentation' }
    ],
    mcpStatus: 'connected',
    a2aStatus: 'active'
  },
  {
    id: 'design_agent',
    name: 'Design Agent',
    type: 'design',
    status: 'waiting',
    progress: 0,
    icon: <Layers className="w-5 h-5" />,
    description: 'Creating system architecture and design specifications',
    tools: ['Figma', 'Uizard', 'Diagram AI'],
    currentTask: 'Waiting for requirements completion',
    logs: [
      { timestamp: '2025-01-23T10:25:00Z', level: 'info', message: 'Design agent initialized' },
      { timestamp: '2025-01-23T10:26:00Z', level: 'info', message: 'Waiting for requirements input' }
    ],
    mcpStatus: 'connected',
    a2aStatus: 'waiting'
  },
  {
    id: 'code_agent',
    name: 'Code Generation Agent',
    type: 'development',
    status: 'idle',
    progress: 0,
    icon: <Code className="w-5 h-5" />,
    description: 'Generating and implementing code based on specifications',
    tools: ['GitHub', 'VS Code', 'ESLint', 'Prettier'],
    currentTask: 'Ready to start development',
    logs: [
      { timestamp: '2025-01-23T10:20:00Z', level: 'info', message: 'Code generation agent ready' }
    ],
    mcpStatus: 'connected',
    a2aStatus: 'idle'
  },
  {
    id: 'quality_agent',
    name: 'Code Quality Agent',
    type: 'code_quality',
    status: 'idle',
    progress: 0,
    icon: <Shield className="w-5 h-5" />,
    description: 'Ensuring code quality and security standards',
    tools: ['SonarQube', 'Checkmarx', 'Codacy'],
    currentTask: 'Standing by for code review',
    logs: [],
    mcpStatus: 'connected',
    a2aStatus: 'idle'
  },
  {
    id: 'test_agent',
    name: 'Testing Agent',
    type: 'testing',
    status: 'idle',
    progress: 0,
    icon: <TestTube className="w-5 h-5" />,
    description: 'Running comprehensive test suites',
    tools: ['Selenium', 'Jest', 'Postman'],
    currentTask: 'Awaiting code completion',
    logs: [],
    mcpStatus: 'connected',
    a2aStatus: 'idle'
  },
  {
    id: 'ci_agent',
    name: 'CI/CD Agent',
    type: 'integration',
    status: 'idle',
    progress: 0,
    icon: <GitBranch className="w-5 h-5" />,
    description: 'Managing continuous integration and deployment',
    tools: ['Jenkins', 'GitHub Actions', 'Azure DevOps'],
    currentTask: 'Pipeline ready',
    logs: [],
    mcpStatus: 'connected',
    a2aStatus: 'idle'
  },
  {
    id: 'deploy_agent',
    name: 'Deployment Agent',
    type: 'deployment',
    status: 'idle',
    progress: 0,
    icon: <Rocket className="w-5 h-5" />,
    description: 'Deploying applications to target environments',
    tools: ['ArgoCD', 'Spinnaker', 'Octopus Deploy'],
    currentTask: 'Deployment ready',
    logs: [],
    mcpStatus: 'connected',
    a2aStatus: 'idle'
  },
  {
    id: 'monitor_agent',
    name: 'Monitoring Agent',
    type: 'monitoring',
    status: 'idle',
    progress: 0,
    icon: <Monitor className="w-5 h-5" />,
    description: 'Setting up monitoring and observability',
    tools: ['Prometheus', 'Grafana', 'Datadog'],
    currentTask: 'Monitoring setup ready',
    logs: [],
    mcpStatus: 'connected',
    a2aStatus: 'idle'
  }
];

// Animated connector component
const AnimatedConnector = ({ from, to, active, theme }) => {
  const isActive = active || from.status === 'running' || from.status === 'completed';
  
  return (
    <div className="relative flex items-center justify-center w-16 h-8">
      <div 
        className={`absolute inset-0 flex items-center justify-center transition-all duration-500 ${
          isActive ? 'opacity-100' : 'opacity-30'
        }`}
        style={{ borderColor: theme.accent }}
      >
        <div className="w-full h-0.5 bg-gradient-to-r from-transparent via-current to-transparent" 
             style={{ color: isActive ? theme.accent : theme.border }}>
        </div>
        {isActive && (
          <div className="absolute w-2 h-2 rounded-full animate-pulse"
               style={{ backgroundColor: theme.accent }}>
          </div>
        )}
        <ChevronRight className={`absolute w-4 h-4 ${isActive ? 'animate-pulse' : ''}`}
                     style={{ color: isActive ? theme.accent : theme.border }} />
      </div>
      {isActive && (
        <div className="absolute w-full h-0.5 bg-gradient-to-r from-transparent via-current to-transparent animate-pulse"
             style={{ color: theme.accent }}>
        </div>
      )}
    </div>
  );
};

// Agent card component
const AgentCard = ({ agent, theme, onViewLogs, isExpanded, onToggleExpand }) => {
  const getStatusColor = (status) => {
    switch (status) {
      case 'running': return theme.accent;
      case 'completed': return theme.success;
      case 'error': return theme.error;
      case 'waiting': return theme.warning;
      default: return theme.textSecondary;
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'running': return <Activity className="w-4 h-4 animate-spin" />;
      case 'completed': return <CheckCircle className="w-4 h-4" />;
      case 'error': return <AlertCircle className="w-4 h-4" />;
      case 'waiting': return <Clock className="w-4 h-4" />;
      default: return <Clock className="w-4 h-4" />;
    }
  };

  return (
    <div 
      className="relative p-4 rounded-lg border transition-all duration-300 hover:shadow-lg cursor-pointer"
      style={{ 
        backgroundColor: theme.cardBg,
        borderColor: agent.status === 'running' ? theme.accent : theme.border,
        boxShadow: agent.status === 'running' ? `0 0 20px ${theme.accent}40` : 'none'
      }}
      onClick={onToggleExpand}
    >
      {/* Status indicator */}
      <div 
        className="absolute top-2 right-2 w-3 h-3 rounded-full animate-pulse"
        style={{ backgroundColor: getStatusColor(agent.status) }}
      />
      
      {/* Agent header */}
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center space-x-3">
          <div 
            className="p-2 rounded-lg"
            style={{ backgroundColor: `${theme.accent}20` }}
          >
            {React.cloneElement(agent.icon, { style: { color: theme.accent } })}
          </div>
          <div>
            <h3 className="font-semibold" style={{ color: theme.text }}>
              {agent.name}
            </h3>
            <p className="text-sm" style={{ color: theme.textSecondary }}>
              {agent.description}
            </p>
          </div>
        </div>
        <div className="flex items-center space-x-2">
          {getStatusIcon(agent.status)}
          {isExpanded ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
        </div>
      </div>

      {/* Progress bar */}
      <div className="mb-3">
        <div className="flex justify-between items-center mb-1">
          <span className="text-sm font-medium" style={{ color: theme.text }}>
            Progress
          </span>
          <span className="text-sm" style={{ color: theme.textSecondary }}>
            {agent.progress}%
          </span>
        </div>
        <div 
          className="w-full h-2 rounded-full"
          style={{ backgroundColor: theme.border }}
        >
          <div 
            className="h-2 rounded-full transition-all duration-500"
            style={{ 
              width: `${agent.progress}%`,
              backgroundColor: theme.accent,
              boxShadow: agent.progress > 0 ? `0 0 10px ${theme.accent}40` : 'none'
            }}
          />
        </div>
      </div>

      {/* Current task */}
      <div className="mb-3">
        <p className="text-sm font-medium" style={{ color: theme.text }}>
          Current Task:
        </p>
        <p className="text-sm" style={{ color: theme.textSecondary }}>
          {agent.currentTask}
        </p>
      </div>

      {/* Protocol status */}
      <div className="flex space-x-4 mb-3">
        <div className="flex items-center space-x-1">
          <div 
            className={`w-2 h-2 rounded-full ${agent.mcpStatus === 'connected' ? 'bg-green-500' : 'bg-red-500'}`}
          />
          <span className="text-xs" style={{ color: theme.textSecondary }}>
            MCP: {agent.mcpStatus}
          </span>
        </div>
        <div className="flex items-center space-x-1">
          <div 
            className={`w-2 h-2 rounded-full ${agent.a2aStatus === 'active' || agent.a2aStatus === 'connected' ? 'bg-green-500' : 'bg-yellow-500'}`}
          />
          <span className="text-xs" style={{ color: theme.textSecondary }}>
            A2A: {agent.a2aStatus}
          </span>
        </div>
      </div>

      {/* Expanded content */}
      {isExpanded && (
        <div className="border-t pt-3 mt-3" style={{ borderColor: theme.border }}>
          {/* Tools */}
          <div className="mb-3">
            <h4 className="text-sm font-medium mb-2" style={{ color: theme.text }}>
              Connected Tools:
            </h4>
            <div className="flex flex-wrap gap-1">
              {agent.tools.map((tool, index) => (
                <span 
                  key={index}
                  className="px-2 py-1 text-xs rounded-full"
                  style={{ 
                    backgroundColor: `${theme.accent}20`,
                    color: theme.accent 
                  }}
                >
                  {tool}
                </span>
              ))}
            </div>
          </div>

          {/* Recent logs */}
          <div className="mb-3">
            <div className="flex justify-between items-center mb-2">
              <h4 className="text-sm font-medium" style={{ color: theme.text }}>
                Recent Activity:
              </h4>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onViewLogs(agent);
                }}
                className="text-xs px-2 py-1 rounded transition-colors"
                style={{ 
                  backgroundColor: `${theme.accent}20`,
                  color: theme.accent 
                }}
              >
                View All Logs
              </button>
            </div>
            <div className="space-y-1 max-h-32 overflow-y-auto">
              {agent.logs.slice(-3).map((log, index) => (
                <div key={index} className="text-xs p-2 rounded" style={{ backgroundColor: theme.secondary }}>
                  <div className="flex justify-between items-start">
                    <span 
                      className={`font-medium ${
                        log.level === 'success' ? 'text-green-500' : 
                        log.level === 'error' ? 'text-red-500' : 
                        log.level === 'warning' ? 'text-yellow-500' : ''
                      }`}
                      style={{ color: log.level === 'info' ? theme.text : undefined }}
                    >
                      {log.level.toUpperCase()}
                    </span>
                    <span style={{ color: theme.textSecondary }}>
                      {new Date(log.timestamp).toLocaleTimeString()}
                    </span>
                  </div>
                  <p style={{ color: theme.textSecondary }}>{log.message}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// Log viewer modal
const LogViewer = ({ agent, theme, onClose }) => {
  const [filter, setFilter] = useState('all');
  
  const filteredLogs = agent.logs.filter(log => 
    filter === 'all' || log.level === filter
  );

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div 
        className="w-full max-w-4xl h-3/4 rounded-lg overflow-hidden"
        style={{ backgroundColor: theme.cardBg }}
      >
        {/* Header */}
        <div className="flex justify-between items-center p-4 border-b" style={{ borderColor: theme.border }}>
          <div>
            <h2 className="text-lg font-semibold" style={{ color: theme.text }}>
              {agent.name} - Execution Logs
            </h2>
            <p className="text-sm" style={{ color: theme.textSecondary }}>
              Real-time agent execution monitoring
            </p>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-opacity-80 transition-colors"
            style={{ backgroundColor: theme.secondary }}
          >
            <Square className="w-5 h-5" style={{ color: theme.text }} />
          </button>
        </div>

        {/* Filters */}
        <div className="p-4 border-b" style={{ borderColor: theme.border }}>
          <div className="flex space-x-2">
            {['all', 'info', 'success', 'warning', 'error'].map(level => (
              <button
                key={level}
                onClick={() => setFilter(level)}
                className={`px-3 py-1 text-sm rounded transition-all ${
                  filter === level ? 'font-medium' : ''
                }`}
                style={{
                  backgroundColor: filter === level ? theme.accent : theme.secondary,
                  color: filter === level ? theme.background : theme.text
                }}
              >
                {level.charAt(0).toUpperCase() + level.slice(1)}
              </button>
            ))}
          </div>
        </div>

        {/* Logs */}
        <div className="flex-1 overflow-y-auto p-4 font-mono text-sm">
          {filteredLogs.length === 0 ? (
            <div className="text-center py-8" style={{ color: theme.textSecondary }}>
              No logs available for the selected filter.
            </div>
          ) : (
            <div className="space-y-2">
              {filteredLogs.map((log, index) => (
                <div 
                  key={index}
                  className="p-3 rounded border-l-4"
                  style={{ 
                    backgroundColor: theme.secondary,
                    borderLeftColor: 
                      log.level === 'success' ? theme.success :
                      log.level === 'error' ? theme.error :
                      log.level === 'warning' ? theme.warning : theme.accent
                  }}
                >
                  <div className="flex justify-between items-start mb-1">
                    <span 
                      className="font-medium text-xs px-2 py-1 rounded"
                      style={{ 
                        backgroundColor: 
                          log.level === 'success' ? `${theme.success}20` :
                          log.level === 'error' ? `${theme.error}20` :
                          log.level === 'warning' ? `${theme.warning}20` : `${theme.accent}20`,
                        color: 
                          log.level === 'success' ? theme.success :
                          log.level === 'error' ? theme.error :
                          log.level === 'warning' ? theme.warning : theme.accent
                      }}
                    >
                      {log.level.toUpperCase()}
                    </span>
                    <span className="text-xs" style={{ color: theme.textSecondary }}>
                      {new Date(log.timestamp).toLocaleString()}
                    </span>
                  </div>
                  <p style={{ color: theme.text }}>{log.message}</p>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

// Main dashboard component
const AgenticSDLCDashboard = () => {
  const [currentTheme, setCurrentTheme] = useState('dark');
  const [agents, setAgents] = useState(mockAgents);
  const [selectedAgent, setSelectedAgent] = useState(null);
  const [expandedAgents, setExpandedAgents] = useState(new Set());
  const [workflowRunning, setWorkflowRunning] = useState(false);
  const [notifications, setNotifications] = useState([]);

  const theme = themes[currentTheme];

  // Simulate real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      setAgents(prevAgents => 
        prevAgents.map(agent => {
          if (agent.status === 'running') {
            const newProgress = Math.min(agent.progress + Math.random() * 5, 100);
            const updatedAgent = { ...agent, progress: newProgress };
            
            // Add new log entry occasionally
            if (Math.random() < 0.3) {
              const logMessages = [
                'Processing data batch',
                'Connecting to external service',
                'Validating requirements',
                'Generating documentation',
                'Running quality checks'
              ];
              const newLog = {
                timestamp: new Date().toISOString(),
                level: Math.random() < 0.8 ? 'info' : 'success',
                message: logMessages[Math.floor(Math.random() * logMessages.length)]
              };
              updatedAgent.logs = [...agent.logs, newLog].slice(-10); // Keep last 10 logs
            }
            
            // Complete agent when progress reaches 100%
            if (newProgress >= 100) {
              updatedAgent.status = 'completed';
              updatedAgent.currentTask = 'Task completed successfully';
            }
            
            return updatedAgent;
          }
          return agent;
        })
      );
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  const handleStartWorkflow = () => {
    setWorkflowRunning(true);
    // Start the first agent
    setAgents(prevAgents => 
      prevAgents.map((agent, index) => 
        index === 0 ? { ...agent, status: 'running' } : agent
      )
    );
  };

  const handleStopWorkflow = () => {
    setWorkflowRunning(false);
    setAgents(prevAgents => 
      prevAgents.map(agent => ({ 
        ...agent, 
        status: agent.status === 'running' ? 'idle' : agent.status 
      }))
    );
  };

  const handleViewLogs = (agent) => {
    setSelectedAgent(agent);
  };

  const handleToggleExpand = (agentId) => {
    setExpandedAgents(prev => {
      const newSet = new Set(prev);
      if (newSet.has(agentId)) {
        newSet.delete(agentId);
      } else {
        newSet.add(agentId);
      }
      return newSet;
    });
  };

  const getWorkflowStats = () => {
    const completed = agents.filter(a => a.status === 'completed').length;
    const running = agents.filter(a => a.status === 'running').length;
    const waiting = agents.filter(a => a.status === 'waiting').length;
    const totalProgress = agents.reduce((sum, agent) => sum + agent.progress, 0) / agents.length;
    
    return { completed, running, waiting, totalProgress };
  };

  const stats = getWorkflowStats();

  return (
    <div 
      className="min-h-screen transition-all duration-500"
      style={{ 
        backgroundColor: theme.background,
        background: `linear-gradient(135deg, ${theme.gradientFrom} 0%, ${theme.gradientTo} 100%)`
      }}
    >
      {/* Header */}
      <div 
        className="sticky top-0 z-40 backdrop-blur-lg border-b"
        style={{ 
          backgroundColor: `${theme.primary}90`,
          borderColor: theme.border 
        }}
      >
        <div className="container mx-auto px-6 py-4">
          <div className="flex justify-between items-center">
            <div>
              <h1 className="text-2xl font-bold" style={{ color: theme.text }}>
                Agentic AI SDLC Pipeline
              </h1>
              <p className="text-sm" style={{ color: theme.textSecondary }}>
                Real-time Software Development Lifecycle Automation
              </p>
            </div>
            
            <div className="flex items-center space-x-4">
              {/* Theme selector */}
              <select
                value={currentTheme}
                onChange={(e) => setCurrentTheme(e.target.value)}
                className="px-3 py-2 rounded-lg border text-sm"
                style={{ 
                  backgroundColor: theme.cardBg,
                  borderColor: theme.border,
                  color: theme.text
                }}
              >
                <option value="dark">Dark Theme</option>
                <option value="light">Light Theme</option>
                <option value="neon">Neon Theme</option>
                <option value="ocean">Ocean Theme</option>
              </select>

              {/* Control buttons */}
              <div className="flex space-x-2">
                {!workflowRunning ? (
                  <button
                    onClick={handleStartWorkflow}
                    className="flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-all hover:scale-105"
                    style={{ 
                      backgroundColor: theme.success,
                      color: theme.background 
                    }}
                  >
                    <Play className="w-4 h-4" />
                    <span>Start Workflow</span>
                  </button>
                ) : (
                  <button
                    onClick={handleStopWorkflow}
                    className="flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-all hover:scale-105"
                    style={{ 
                      backgroundColor: theme.error,
                      color: theme.background 
                    }}
                  >
                    <Square className="w-4 h-4" />
                    <span>Stop Workflow</span>
                  </button>
                )}
                
                <button className="p-2 rounded-lg transition-colors" style={{ backgroundColor: theme.secondary }}>
                  <Settings className="w-5 h-5" style={{ color: theme.text }} />
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Stats bar */}
      <div className="container mx-auto px-6 py-4">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div 
            className="p-4 rounded-lg border"
            style={{ backgroundColor: theme.cardBg, borderColor: theme.border }}
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm" style={{ color: theme.textSecondary }}>Overall Progress</p>
                <p className="text-2xl font-bold" style={{ color: theme.text }}>
                  {Math.round(stats.totalProgress)}%
                </p>
              </div>
              <BarChart3 className="w-8 h-8" style={{ color: theme.accent }} />
            </div>
          </div>
          
          <div 
            className="p-4 rounded-lg border"
            style={{ backgroundColor: theme.cardBg, borderColor: theme.border }}
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm" style={{ color: theme.textSecondary }}>Running Agents</p>
                <p className="text-2xl font-bold" style={{ color: theme.accent }}>
                  {stats.running}
                </p>
              </div>
              <Activity className="w-8 h-8 animate-pulse" style={{ color: theme.accent }} />
            </div>
          </div>
          
          <div 
            className="p-4 rounded-lg border"
            style={{ backgroundColor: theme.cardBg, borderColor: theme.border }}
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm" style={{ color: theme.textSecondary }}>Completed</p>
                <p className="text-2xl font-bold" style={{ color: theme.success }}>
                  {stats.completed}
                </p>
              </div>
              <CheckCircle className="w-8 h-8" style={{ color: theme.success }} />
            </div>
          </div>
          
          <div 
            className="p-4 rounded-lg border"
            style={{ backgroundColor: theme.cardBg, borderColor: theme.border }}
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm" style={{ color: theme.textSecondary }}>Waiting</p>
                <p className="text-2xl font-bold" style={{ color: theme.warning }}>
                  {stats.waiting}
                </p>
              </div>
              <Clock className="w-8 h-8" style={{ color: theme.warning }} />
            </div>
          </div>
        </div>
      </div>

      {/* Workflow visualization */}
      <div className="container mx-auto px-6 pb-8">
        <div className="flex flex-wrap justify-center items-center gap-4">
          {agents.map((agent, index) => (
            <React.Fragment key={agent.id}>
              <div className="flex-shrink-0">
                <AgentCard
                  agent={agent}
                  theme={theme}
                  onViewLogs={handleViewLogs}
                  isExpanded={expandedAgents.has(agent.id)}
                  onToggleExpand={() => handleToggleExpand(agent.id)}
                />
              </div>
              
              {index < agents.length - 1 && (
                <AnimatedConnector
                  from={agent}
                  to={agents[index + 1]}
                  active={workflowRunning}
                  theme={theme}
                />
              )}
            </React.Fragment>
          ))}
        </div>
      </div>

      {/* Log viewer modal */}
      {selectedAgent && (
        <LogViewer
          agent={selectedAgent}
          theme={theme}
          onClose={() => setSelectedAgent(null)}
        />
      )}
    </div>
  );
};

export default AgenticSDLCDashboard; 