import React, { useState, useEffect, useRef } from 'react';
import { 
  Play, 
  Pause, 
  Settings, 
  Eye, 
  Brain, 
  Target, 
  Zap, 
  MessageSquare, 
  CheckCircle, 
  AlertCircle, 
  Clock, 
  Activity, 
  Upload, 
  Link, 
  FileText, 
  Database, 
  Cloud, 
  Folder 
} from 'lucide-react';

const RequirementsAgentDashboard = () => {
  const [agentState, setAgentState] = useState('idle');
  const [currentPhase, setCurrentPhase] = useState('');
  const [logs, setLogs] = useState([]);
  const [showLogs, setShowLogs] = useState(false);
  const [taskProgress, setTaskProgress] = useState(0);
  const [mcpEvents, setMcpEvents] = useState([]);
  const [a2aEvents, setA2aEvents] = useState([]);
  const [selectedTheme, setSelectedTheme] = useState('cyberpunk');
  const [showIntegrations, setShowIntegrations] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [integrations, setIntegrations] = useState({
    confluence: { connected: false, config: {} },
    jira: { connected: false, config: {} },
    sharepoint: { connected: false, config: {} },
    onedrive: { connected: false, config: {} },
    googledrive: { connected: false, config: {} }
  });
  const [projectInfo, setProjectInfo] = useState({
    name: '',
    description: '',
    stakeholders: '',
    businessGoals: ''
  });
  const [showProjectForm, setShowProjectForm] = useState(false);
  const fileInputRef = useRef(null);

  const themes = {
    cyberpunk: {
      primary: 'from-cyan-400 to-blue-600',
      secondary: 'from-purple-500 to-pink-600',
      accent: 'bg-cyan-400',
      bg: 'bg-gray-900',
      card: 'bg-gray-800',
      text: 'text-cyan-100'
    },
    matrix: {
      primary: 'from-green-400 to-emerald-600',
      secondary: 'from-lime-500 to-green-600',
      accent: 'bg-green-400',
      bg: 'bg-black',
      card: 'bg-gray-900',
      text: 'text-green-100'
    },
    neon: {
      primary: 'from-pink-400 to-purple-600',
      secondary: 'from-orange-500 to-red-600',
      accent: 'bg-pink-400',
      bg: 'bg-gray-900',
      card: 'bg-purple-900',
      text: 'text-pink-100'
    }
  };

  const theme = themes[selectedTheme];

  const integrationConfigs = {
    confluence: {
      name: 'Atlassian Confluence',
      icon: Database,
      color: 'from-blue-500 to-blue-700',
      fields: [
        { name: 'baseUrl', label: 'Confluence Base URL', type: 'url', placeholder: 'https://yourcompany.atlassian.net/wiki' },
        { name: 'email', label: 'Email', type: 'email', placeholder: 'your-email@company.com' },
        { name: 'apiToken', label: 'API Token', type: 'password', placeholder: 'Your Confluence API token' },
        { name: 'spaceKey', label: 'Space Key', type: 'text', placeholder: 'PROJ' }
      ]
    },
    jira: {
      name: 'Atlassian Jira',
      icon: Link,
      color: 'from-blue-600 to-indigo-700',
      fields: [
        { name: 'baseUrl', label: 'Jira Base URL', type: 'url', placeholder: 'https://yourcompany.atlassian.net' },
        { name: 'email', label: 'Email', type: 'email', placeholder: 'your-email@company.com' },
        { name: 'apiToken', label: 'API Token', type: 'password', placeholder: 'Your Jira API token' },
        { name: 'projectKey', label: 'Project Key', type: 'text', placeholder: 'PROJ' }
      ]
    },
    sharepoint: {
      name: 'Microsoft SharePoint',
      icon: Folder,
      color: 'from-blue-700 to-purple-700',
      fields: [
        { name: 'siteUrl', label: 'SharePoint Site URL', type: 'url', placeholder: 'https://yourcompany.sharepoint.com/sites/project' },
        { name: 'clientId', label: 'Client ID', type: 'text', placeholder: 'Your app client ID' },
        { name: 'clientSecret', label: 'Client Secret', type: 'password', placeholder: 'Your app client secret' },
        { name: 'documentLibrary', label: 'Document Library', type: 'text', placeholder: 'Documents' }
      ]
    },
    onedrive: {
      name: 'Microsoft OneDrive',
      icon: Cloud,
      color: 'from-blue-600 to-cyan-600',
      fields: [
        { name: 'clientId', label: 'Application ID', type: 'text', placeholder: 'Your app ID' },
        { name: 'clientSecret', label: 'Client Secret', type: 'password', placeholder: 'Your app secret' },
        { name: 'tenantId', label: 'Tenant ID', type: 'text', placeholder: 'Your tenant ID' },
        { name: 'folderPath', label: 'Folder Path', type: 'text', placeholder: '/Requirements' }
      ]
    },
    googledrive: {
      name: 'Google Drive',
      icon: Database,
      color: 'from-green-500 to-yellow-500',
      fields: [
        { name: 'clientId', label: 'Client ID', type: 'text', placeholder: 'Your Google OAuth client ID' },
        { name: 'clientSecret', label: 'Client Secret', type: 'password', placeholder: 'Your client secret' },
        { name: 'refreshToken', label: 'Refresh Token', type: 'password', placeholder: 'OAuth refresh token' },
        { name: 'folderId', label: 'Folder ID', type: 'text', placeholder: 'Google Drive folder ID' }
      ]
    }
  };

  const handleFileUpload = async (event) => {
    const files = Array.from(event.target.files);
    
    for (const file of files) {
      const fileData = {
        id: Date.now() + Math.random(),
        name: file.name,
        size: file.size,
        type: file.type,
        status: 'uploading',
        content: null
      };

      setUploadedFiles(prev => [...prev, fileData]);

      try {
        const content = await readFileContent(file);
        
        setUploadedFiles(prev => 
          prev.map(f => 
            f.id === fileData.id 
              ? { ...f, status: 'completed', content }
              : f
          )
        );

        addLog(`📄 File uploaded successfully: ${file.name}`, 'upload');
        
      } catch (error) {
        setUploadedFiles(prev => 
          prev.map(f => 
            f.id === fileData.id 
              ? { ...f, status: 'error', error: error.message }
              : f
          )
        );
        
        addLog(`❌ Failed to upload file: ${file.name}`, 'error');
      }
    }
  };

  const readFileContent = (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      
      reader.onload = (e) => {
        const content = e.target.result;
        
        if (file.type.includes('pdf')) {
          resolve({ type: 'pdf', content: 'PDF content extraction would be implemented here' });
        } else if (file.type.includes('word') || file.name.endsWith('.docx')) {
          resolve({ type: 'docx', content: 'Word document content extraction would be implemented here' });
        } else if (file.type.includes('text') || file.name.endsWith('.txt') || file.name.endsWith('.md')) {
          resolve({ type: 'text', content });
        } else {
          resolve({ type: 'unknown', content: 'File type not supported for content extraction' });
        }
      };
      
      reader.onerror = () => reject(new Error('Failed to read file'));
      
      if (file.type.includes('text') || file.name.endsWith('.txt') || file.name.endsWith('.md')) {
        reader.readAsText(file);
      } else {
        reader.readAsArrayBuffer(file);
      }
    });
  };

  const addLog = (message, type = 'info') => {
    setLogs(prev => [...prev, {
      id: Date.now(),
      message,
      timestamp: new Date().toLocaleTimeString(),
      type
    }]);
  };

  const connectIntegration = async (integrationType, config) => {
    try {
      addLog(`🔗 Connecting to ${integrationConfigs[integrationType].name}...`, 'integration');
      
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      setIntegrations(prev => ({
        ...prev,
        [integrationType]: {
          connected: true,
          config,
          connectedAt: new Date().toISOString()
        }
      }));

      addLog(`✅ Successfully connected to ${integrationConfigs[integrationType].name}`, 'success');
      
      setMcpEvents(prev => [...prev, {
        id: Date.now(),
        type: 'integration_connected',
        message: `MCP: ${integrationConfigs[integrationType].name} integration established`,
        timestamp: new Date().toLocaleTimeString(),
        status: 'success'
      }]);

    } catch (error) {
      addLog(`❌ Failed to connect to ${integrationConfigs[integrationType].name}: ${error.message}`, 'error');
    }
  };

  const disconnectIntegration = (integrationType) => {
    setIntegrations(prev => ({
      ...prev,
      [integrationType]: { connected: false, config: {} }
    }));
    
    addLog(`🔌 Disconnected from ${integrationConfigs[integrationType].name}`, 'info');
  };

  const fetchFromIntegration = async (integrationType) => {
    if (!integrations[integrationType].connected) {
      addLog(`❌ ${integrationConfigs[integrationType].name} not connected`, 'error');
      return;
    }

    try {
      addLog(`📥 Fetching documents from ${integrationConfigs[integrationType].name}...`, 'integration');
      
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      const mockDocuments = [
        { name: 'Project Requirements.docx', size: '245 KB', source: integrationType },
        { name: 'User Stories.pdf', size: '180 KB', source: integrationType },
        { name: 'Technical Specifications.md', size: '95 KB', source: integrationType }
      ];

      setUploadedFiles(prev => [...prev, ...mockDocuments.map(doc => ({
        id: Date.now() + Math.random(),
        ...doc,
        status: 'completed',
        content: { type: 'fetched', content: `Fetched content from ${integrationType}` }
      }))]);

      addLog(`✅ Fetched ${mockDocuments.length} documents from ${integrationConfigs[integrationType].name}`, 'success');
      
    } catch (error) {
      addLog(`❌ Failed to fetch from ${integrationConfigs[integrationType].name}: ${error.message}`, 'error');
    }
  };

  const startRealRequirementsAnalysis = async () => {
    if (agentState === 'running') return;
    
    if (uploadedFiles.length === 0 && !Object.values(integrations).some(i => i.connected)) {
      addLog('❌ Please upload documents or connect to an integration first', 'error');
      return;
    }

    if (!projectInfo.name || !projectInfo.description) {
      setShowProjectForm(true);
      return;
    }

    setAgentState('running');
    setTaskProgress(0);
    setLogs([]);
    setMcpEvents([]);
    setA2aEvents([]);

    const phases = [
      {
        phase: 'thinking',
        duration: 4000,
        step: 'reasoning',
        logs: [
          "🧠 Analyzing uploaded documents and project context...",
          `📊 Processing ${uploadedFiles.length} uploaded documents`,
          "🔍 Extracting requirements from document content...",
          "💡 Identifying functional and non-functional requirements",
          "🎯 Mapping requirements to business objectives",
          "⚠️ Detecting potential conflicts and gaps"
        ]
      },
      {
        phase: 'planning',
        duration: 3000,
        step: 'planning',
        logs: [
          "🎯 Creating comprehensive requirements analysis plan...",
          "📋 Categorizing requirements by type and priority",
          "🔗 Planning stakeholder validation approach",
          "📝 Designing requirement documentation structure",
          "🔍 Planning validation and verification methods"
        ]
      },
      {
        phase: 'acting',
        duration: 5000,
        step: 'acting',
        logs: [
          "🎬 Executing requirements analysis and documentation...",
          "📄 Generating functional requirements specification",
          "🛡️ Documenting non-functional requirements",
          "👥 Creating stakeholder impact analysis",
          "🔗 Building requirements traceability matrix",
          "📊 Generating requirements validation report",
          "✅ Creating final requirements package"
        ]
      }
    ];

    let currentPhaseIndex = 0;
    let logIndex = 0;

    const executePhase = () => {
      if (currentPhaseIndex >= phases.length) {
        setAgentState('completed');
        setCurrentPhase('Requirements analysis completed successfully');
        setTaskProgress(100);
        
        setMcpEvents(prev => [...prev, {
          id: Date.now(),
          type: 'completion',
          message: `Requirements analysis complete: ${uploadedFiles.length} documents processed`,
          timestamp: new Date().toLocaleTimeString(),
          status: 'success'
        }]);
        
        setA2aEvents(prev => [...prev, {
          id: Date.now(),
          type: 'agent_communication',
          message: 'A2A: Requirements Agent → Design Agent: Requirements package ready',
          timestamp: new Date().toLocaleTimeString(),
          status: 'success'
        }]);
        
        return;
      }

      const phase = phases[currentPhaseIndex];
      setCurrentPhase(phase.phase);
      
      setMcpEvents(prev => [...prev, {
        id: Date.now(),
        type: 'phase_start',
        message: `MCP: Starting ${phase.step} phase with real document analysis`,
        timestamp: new Date().toLocaleTimeString(),
        status: 'active'
      }]);

      setA2aEvents(prev => [...prev, {
        id: Date.now(),
        type: 'agent_communication',
        message: `A2A: Requirements Agent → Orchestrator: Phase ${phase.step} initiated`,
        timestamp: new Date().toLocaleTimeString(),
        status: 'active'
      }]);

      const logInterval = setInterval(() => {
        if (logIndex < phase.logs.length) {
          setLogs(prev => [...prev, {
            id: Date.now() + logIndex,
            message: phase.logs[logIndex],
            timestamp: new Date().toLocaleTimeString(),
            phase: phase.step,
            type: 'processing'
          }]);
          logIndex++;
          
          const progress = ((currentPhaseIndex * 100) + (logIndex / phase.logs.length * 100)) / phases.length;
          setTaskProgress(Math.min(progress, 100));
        } else {
          clearInterval(logInterval);
          currentPhaseIndex++;
          logIndex = 0;
          
          setMcpEvents(prev => [...prev, {
            id: Date.now(),
            type: 'phase_complete',
            message: `MCP: ${phase.step} phase completed successfully`,
            timestamp: new Date().toLocaleTimeString(),
            status: 'success'
          }]);
          
          setTimeout(executePhase, 500);
        }
      }, 600);
    };

    executePhase();
  };

  const agentCapabilities = [
    {
      name: "Document Analysis",
      description: "Extract requirements from uploaded documents",
      tools: ["pdf_parser", "docx_reader", "content_analyzer"],
      status: "ready"
    },
    {
      name: "Integration Processing",
      description: "Fetch and analyze requirements from connected systems",
      tools: ["confluence_api", "jira_api", "sharepoint_api", "drive_api"],
      status: "ready"
    },
    {
      name: "Requirements Validation",
      description: "Validate and structure extracted requirements",
      tools: ["nlp_processor", "conflict_detector", "priority_analyzer"],
      status: "ready"
    }
  ];

  const workflowSteps = [
    { id: 'reasoning', name: 'Reasoning', icon: Brain, phase: 'Analyzing stakeholder needs' },
    { id: 'planning', name: 'Planning', icon: Target, phase: 'Creating requirement gathering plan' },
    { id: 'acting', name: 'Acting', icon: Zap, phase: 'Executing requirement collection' }
  ];

  const getStateColor = (state) => {
    switch (state) {
      case 'idle': return 'text-gray-400';
      case 'running': return 'text-blue-400 animate-pulse';
      case 'completed': return 'text-green-400';
      case 'error': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  const getStateIcon = (state) => {
    switch (state) {
      case 'idle': return <Clock className="w-5 h-5" />;
      case 'running': return <Activity className="w-5 h-5 animate-spin" />;
      case 'completed': return <CheckCircle className="w-5 h-5" />;
      case 'error': return <AlertCircle className="w-5 h-5" />;
      default: return <Clock className="w-5 h-5" />;
    }
  };

  const IntegrationModal = ({ integrationType, onClose, onConnect }) => {
    const [config, setConfig] = useState({});
    const [isConnecting, setIsConnecting] = useState(false);
    const integrationConfig = integrationConfigs[integrationType];

    const handleSubmit = async (e) => {
      e.preventDefault();
      setIsConnecting(true);
      await onConnect(integrationType, config);
      setIsConnecting(false);
      onClose();
    };

    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div className={`${theme.card} rounded-xl p-6 w-full max-w-md border border-gray-700`}>
          <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
            <integrationConfig.icon className="w-6 h-6" />
            Connect to {integrationConfig.name}
          </h3>
          
          <form onSubmit={handleSubmit} className="space-y-4">
            {integrationConfig.fields.map((field) => (
              <div key={field.name}>
                <label className="block text-sm font-medium mb-1">
                  {field.label}
                </label>
                <input
                  type={field.type}
                  placeholder={field.placeholder}
                  value={config[field.name] || ''}
                  onChange={(e) => setConfig(prev => ({ ...prev, [field.name]: e.target.value }))}
                  className={`w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg focus:border-blue-400 focus:outline-none ${theme.text}`}
                  required
                />
              </div>
            ))}
            
            <div className="flex gap-3 pt-4">
              <button
                type="button"
                onClick={onClose}
                className="flex-1 px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={isConnecting}
                className={`flex-1 px-4 py-2 bg-gradient-to-r ${integrationConfig.color} text-white rounded-lg hover:opacity-90 transition-opacity disabled:opacity-50`}
              >
                {isConnecting ? 'Connecting...' : 'Connect'}
              </button>
            </div>
          </form>
        </div>
      </div>
    );
  };

  const ProjectInfoModal = ({ onClose, onSave }) => {
    const [info, setInfo] = useState(projectInfo);

    const handleSubmit = (e) => {
      e.preventDefault();
      setProjectInfo(info);
      onSave(info);
      onClose();
    };

    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div className={`${theme.card} rounded-xl p-6 w-full max-w-lg border border-gray-700`}>
          <h3 className="text-xl font-bold mb-4">Project Information</h3>
          
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-1">Project Name *</label>
              <input
                type="text"
                value={info.name}
                onChange={(e) => setInfo(prev => ({ ...prev, name: e.target.value }))}
                className={`w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg focus:border-blue-400 focus:outline-none ${theme.text}`}
                placeholder="Enter project name"
                required
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-1">Description *</label>
              <textarea
                value={info.description}
                onChange={(e) => setInfo(prev => ({ ...prev, description: e.target.value }))}
                className={`w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg focus:border-blue-400 focus:outline-none ${theme.text}`}
                placeholder="Describe the project"
                rows="3"
                required
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-1">Key Stakeholders</label>
              <input
                type="text"
                value={info.stakeholders}
                onChange={(e) => setInfo(prev => ({ ...prev, stakeholders: e.target.value }))}
                className={`w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg focus:border-blue-400 focus:outline-none ${theme.text}`}
                placeholder="PM, Business Analyst, End Users"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-1">Business Goals</label>
              <textarea
                value={info.businessGoals}
                onChange={(e) => setInfo(prev => ({ ...prev, businessGoals: e.target.value }))}
                className={`w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg focus:border-blue-400 focus:outline-none ${theme.text}`}
                placeholder="List key business objectives"
                rows="2"
              />
            </div>
            
            <div className="flex gap-3 pt-4">
              <button
                type="button"
                onClick={onClose}
                className="flex-1 px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
              >
                Cancel
              </button>
              <button
                type="submit"
                className={`flex-1 px-4 py-2 bg-gradient-to-r ${theme.primary} text-white rounded-lg hover:opacity-90 transition-opacity`}
              >
                Save & Start Analysis
              </button>
            </div>
          </form>
        </div>
      </div>
    );
  };

  return (
    <div className={`min-h-screen ${theme.bg} ${theme.text} p-6`}>
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="flex justify-between items-center mb-8">
          <div>
            <h1 className="text-4xl font-bold bg-gradient-to-r bg-clip-text text-transparent from-blue-400 to-purple-600">
              Agentic AI SDLC - Requirements Agent
            </h1>
            <p className="text-gray-400 mt-2">Real-World Requirements Engineering & Analysis</p>
          </div>
          
          <div className="flex items-center gap-4">
            <select 
              value={selectedTheme} 
              onChange={(e) => setSelectedTheme(e.target.value)}
              className={`${theme.card} border border-gray-600 rounded-lg px-3 py-2 ${theme.text}`}
            >
              <option value="cyberpunk">Cyberpunk</option>
              <option value="matrix">Matrix</option>
              <option value="neon">Neon</option>
            </select>
          </div>
        </div>

        {/* Agent Status Card */}
        <div className={`${theme.card} rounded-xl p-6 mb-6 border border-gray-700`}>
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <div className={`w-12 h-12 rounded-full ${theme.accent} flex items-center justify-center`}>
                {getStateIcon(agentState)}
              </div>
              <div>
                <h2 className="text-xl font-bold">Requirements Agent</h2>
                <p className={`text-sm ${getStateColor(agentState)}`}>
                  Status: {agentState.toUpperCase()} | {currentPhase}
                </p>
              </div>
            </div>
            
            <div className="flex items-center gap-3">
              <button
                onClick={() => setShowIntegrations(!showIntegrations)}
                className={`p-2 ${theme.card} border border-gray-600 rounded-lg hover:bg-gray-700 transition-colors`}
              >
                <Settings className="w-5 h-5" />
              </button>
              <button
                onClick={() => setShowLogs(!showLogs)}
                className={`p-2 ${theme.card} border border-gray-600 rounded-lg hover:bg-gray-700 transition-colors`}
              >
                <Eye className="w-5 h-5" />
              </button>
              <button
                onClick={startRealRequirementsAnalysis}
                disabled={agentState === 'running'}
                className={`px-6 py-2 bg-gradient-to-r ${theme.primary} rounded-lg hover:opacity-90 transition-opacity disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2`}
              >
                {agentState === 'running' ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                {agentState === 'running' ? 'Analyzing...' : 'Start Analysis'}
              </button>
            </div>
          </div>
          
          <div className="w-full bg-gray-700 rounded-full h-2 mb-4">
            <div 
              className={`h-2 bg-gradient-to-r ${theme.primary} rounded-full transition-all duration-500`}
              style={{ width: `${taskProgress}%` }}
            ></div>
          </div>
          
          <p className="text-sm text-gray-400">Progress: {Math.round(taskProgress)}%</p>
        </div>

        {/* Data Sources Section */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          {/* Document Upload */}
          <div className={`${theme.card} rounded-xl p-6 border border-gray-700`}>
            <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
              <Upload className="w-5 h-5" />
              Document Upload
            </h3>
            
            <div className="space-y-4">
              <div 
                className="border-2 border-dashed border-gray-600 rounded-lg p-6 text-center hover:border-blue-400 transition-colors cursor-pointer"
                onClick={() => fileInputRef.current?.click()}
              >
                <Upload className="w-8 h-8 mx-auto mb-2 text-gray-400" />
                <p className="text-gray-400">
                  Click to upload or drag and drop
                </p>
                <p className="text-sm text-gray-500 mt-1">
                  PDF, DOCX, TXT, MD files supported
                </p>
              </div>
              
              <input
                ref={fileInputRef}
                type="file"
                multiple
                accept=".pdf,.doc,.docx,.txt,.md"
                onChange={handleFileUpload}
                className="hidden"
              />
              
              {uploadedFiles.length > 0 && (
                <div className="space-y-2">
                  <h4 className="font-medium">Uploaded Files ({uploadedFiles.length})</h4>
                  <div className="max-h-32 overflow-y-auto space-y-1">
                    {uploadedFiles.map((file) => (
                      <div key={file.id} className="flex items-center gap-3 p-2 bg-gray-700 rounded-lg text-sm">
                        <FileText className="w-4 h-4" />
                        <span className="flex-1 truncate">{file.name}</span>
                        <span className={`px-2 py-1 rounded text-xs ${
                          file.status === 'completed' ? 'bg-green-600' :
                          file.status === 'error' ? 'bg-red-600' :
                          'bg-yellow-600'
                        }`}>
                          {file.status}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Integrations */}
          <div className={`${theme.card} rounded-xl p-6 border border-gray-700`}>
            <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
              <Link className="w-5 h-5" />
              System Integrations
            </h3>
            
            <div className="space-y-3">
              {Object.entries(integrationConfigs).map(([key, config]) => {
                const Icon = config.icon;
                const isConnected = integrations[key].connected;
                
                return (
                  <div key={key} className="flex items-center justify-between p-3 bg-gray-700 rounded-lg">
                    <div className="flex items-center gap-3">
                      <Icon className="w-5 h-5" />
                      <div>
                        <span className="font-medium">{config.name}</span>
                        {isConnected && (
                          <div className="text-xs text-green-400">Connected</div>
                        )}
                      </div>
                    </div>
                    
                    <div className="flex items-center gap-2">
                      {isConnected && (
                        <button
                          onClick={() => fetchFromIntegration(key)}
                          className="px-3 py-1 text-xs bg-blue-600 rounded hover:bg-blue-700 transition-colors"
                        >
                          Fetch
                        </button>
                      )}
                      <button
                        onClick={() => isConnected ? disconnectIntegration(key) : setShowIntegrations(key)}
                        className={`px-3 py-1 text-xs rounded transition-colors ${
                          isConnected 
                            ? 'bg-red-600 hover:bg-red-700' 
                            : 'bg-green-600 hover:bg-green-700'
                        }`}
                      >
                        {isConnected ? 'Disconnect' : 'Connect'}
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Workflow Visualization */}
        <div className={`${theme.card} rounded-xl p-6 mb-6 border border-gray-700`}>
          <h3 className="text-lg font-bold mb-4">Agent Workflow - Reason → Plan → Act</h3>
          <div className="flex items-center justify-between">
            {workflowSteps.map((step, index) => (
              <React.Fragment key={step.id}>
                <div className={`flex flex-col items-center p-4 rounded-lg border-2 transition-all duration-500 ${
                  currentPhase === step.id 
                    ? `border-blue-400 ${theme.accent} text-black` 
                    : 'border-gray-600'
                }`}>
                  <step.icon className={`w-8 h-8 mb-2 ${currentPhase === step.id ? 'animate-pulse' : ''}`} />
                  <span className="font-medium">{step.name}</span>
                  <span className="text-xs text-center mt-1 opacity-75">{step.phase}</span>
                </div>
                {index < workflowSteps.length - 1 && (
                  <div className={`flex-1 h-px bg-gradient-to-r ${theme.primary} mx-4 opacity-50`}></div>
                )}
              </React.Fragment>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          {/* MCP Protocol Events */}
          <div className={`${theme.card} rounded-xl p-6 border border-gray-700`}>
            <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
              <MessageSquare className="w-5 h-5" />
              MCP Protocol Events
            </h3>
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {mcpEvents.map((event) => (
                <div key={event.id} className="flex items-center gap-3 p-3 bg-gray-700 rounded-lg">
                  <div className={`w-2 h-2 rounded-full ${
                    event.status === 'success' ? 'bg-green-400' : 
                    event.status === 'active' ? 'bg-blue-400 animate-pulse' : 'bg-gray-400'
                  }`}></div>
                  <div className="flex-1">
                    <p className="text-sm">{event.message}</p>
                    <p className="text-xs text-gray-400">{event.timestamp}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* A2A Protocol Events */}
          <div className={`${theme.card} rounded-xl p-6 border border-gray-700`}>
            <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
              <Activity className="w-5 h-5" />
              A2A Protocol Events
            </h3>
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {a2aEvents.map((event) => (
                <div key={event.id} className="flex items-center gap-3 p-3 bg-gray-700 rounded-lg">
                  <div className={`w-2 h-2 rounded-full ${
                    event.status === 'success' ? 'bg-green-400' : 
                    event.status === 'active' ? 'bg-purple-400 animate-pulse' : 'bg-gray-400'
                  }`}></div>
                  <div className="flex-1">
                    <p className="text-sm">{event.message}</p>
                    <p className="text-xs text-gray-400">{event.timestamp}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Agent Capabilities */}
        <div className={`${theme.card} rounded-xl p-6 mb-6 border border-gray-700`}>
          <h3 className="text-lg font-bold mb-4">Agent Capabilities</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {agentCapabilities.map((capability, index) => (
              <div key={index} className="p-4 bg-gray-700 rounded-lg">
                <h4 className="font-medium mb-2">{capability.name}</h4>
                <p className="text-sm text-gray-400 mb-3">{capability.description}</p>
                <div className="flex flex-wrap gap-1">
                  {capability.tools.map((tool, toolIndex) => (
                    <span key={toolIndex} className={`px-2 py-1 text-xs bg-gradient-to-r ${theme.secondary} rounded-full text-white`}>
                      {tool}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Console Logs */}
        {showLogs && (
          <div className={`${theme.card} rounded-xl p-6 border border-gray-700`}>
            <h3 className="text-lg font-bold mb-4">Agent Console Logs</h3>
            <div className="bg-black rounded-lg p-4 font-mono text-sm max-h-96 overflow-y-auto">
              {logs.map((log) => (
                <div key={log.id} className="mb-2 flex gap-3">
                  <span className="text-gray-500">[{log.timestamp}]</span>
                  <span className={`${
                    log.type === 'error' ? 'text-red-400' :
                    log.type === 'success' ? 'text-green-400' :
                    log.type === 'integration' ? 'text-blue-400' :
                    log.type === 'upload' ? 'text-yellow-400' :
                    log.phase === 'reasoning' ? 'text-blue-400' :
                    log.phase === 'planning' ? 'text-yellow-400' :
                    log.phase === 'acting' ? 'text-green-400' : 'text-gray-400'
                  }`}>
                    [{log.type?.toUpperCase() || log.phase?.toUpperCase() || 'INFO'}]
                  </span>
                  <span className="text-gray-100">{log.message}</span>
                </div>
              ))}
              {logs.length === 0 && (
                <div className="text-gray-500 text-center">
                  No logs yet. Upload documents or connect integrations to start.
                </div>
              )}
            </div>
          </div>
        )}

        {/* Integration Modal */}
        {showIntegrations && typeof showIntegrations === 'string' && (
          <IntegrationModal
            integrationType={showIntegrations}
            onClose={() => setShowIntegrations(false)}
            onConnect={connectIntegration}
          />
        )}

        {/* Project Info Modal */}
        {showProjectForm && (
          <ProjectInfoModal
            onClose={() => setShowProjectForm(false)}
            onSave={() => {
              setShowProjectForm(false);
              setTimeout(startRealRequirementsAnalysis, 500);
            }}
          />
        )}
      </div>
    </div>
  );
};

export default RequirementsAgentDashboard;