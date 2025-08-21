import React, { useState, useEffect, useRef } from 'react';
import { 
  Upload, 
  FileText, 
  Database, 
  Zap, 
  Settings, 
  Play, 
  Download,
  ChevronRight,
  Loader2,
  CheckCircle,
  AlertCircle,
  Plus,
  X
} from 'lucide-react';

const RAGUserStoriesApp = () => {
  const [activeStep, setActiveStep] = useState(1);
  const [isProcessing, setIsProcessing] = useState(false);
  const [connectorAnimations, setConnectorAnimations] = useState({});
  const [dataSources, setDataSources] = useState({
    jira: { connected: false, status: 'disconnected' },
    confluence: { connected: false, status: 'disconnected' },
    sharepoint: { connected: false, status: 'disconnected' },
    uploads: { files: [], status: 'ready' }
  });
  const [llmConfig, setLlmConfig] = useState({
    provider: 'openai',
    model: 'gpt-4',
    apiKey: '',
    temperature: 0.7
  });
  const [generatedStories, setGeneratedStories] = useState([]);
  const canvasRef = useRef(null);

  // Animated connector system
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const resizeCanvas = () => {
      canvas.width = canvas.offsetWidth;
      canvas.height = canvas.offsetHeight;
    };
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    let animationFrame;
    let time = 0;

    const drawConnector = (startX, startY, endX, endY, active = false) => {
      const gradient = ctx.createLinearGradient(startX, startY, endX, endY);
      
      if (active) {
        // Animated flowing gradient
        const offset = (Math.sin(time * 0.02) + 1) * 0.5;
        gradient.addColorStop(0, `rgba(139, 92, 246, ${0.3 + offset * 0.7})`);
        gradient.addColorStop(0.5, `rgba(79, 70, 229, ${0.5 + offset * 0.5})`);
        gradient.addColorStop(1, `rgba(59, 130, 246, ${0.3 + offset * 0.7})`);
      } else {
        gradient.addColorStop(0, 'rgba(100, 116, 139, 0.2)');
        gradient.addColorStop(1, 'rgba(100, 116, 139, 0.1)');
      }

      ctx.strokeStyle = gradient;
      ctx.lineWidth = active ? 3 : 2;
      ctx.lineCap = 'round';

      // Draw curved connector
      ctx.beginPath();
      ctx.moveTo(startX, startY);
      
      const midX = (startX + endX) / 2;
      const midY = (startY + endY) / 2;
      const curve = 50;
      
      ctx.quadraticCurveTo(midX, midY - curve, endX, endY);
      ctx.stroke();

      // Draw flow particles if active
      if (active) {
        const particlePos = (time * 0.01) % 1;
        const particleX = startX + (endX - startX) * particlePos;
        const particleY = startY + (endY - startY) * particlePos - curve * Math.sin(Math.PI * particlePos);
        
        ctx.fillStyle = 'rgba(139, 92, 246, 0.8)';
        ctx.beginPath();
        ctx.arc(particleX, particleY, 4, 0, Math.PI * 2);
        ctx.fill();
      }
    };

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      time++;

      // Draw connectors based on current state
      const centerY = canvas.height / 2;
      const sourceY = centerY - 100;
      const outputY = centerY + 100;

      // Data sources to processing
      drawConnector(150, sourceY, 300, centerY, isProcessing || activeStep >= 2);
      drawConnector(150, sourceY + 40, 300, centerY, isProcessing || activeStep >= 2);
      drawConnector(150, sourceY + 80, 300, centerY, isProcessing || activeStep >= 2);

      // Processing to LLM
      drawConnector(400, centerY, 550, centerY, isProcessing || activeStep >= 3);

      // LLM to output
      drawConnector(650, centerY, 800, outputY, activeStep >= 4);

      animationFrame = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      window.removeEventListener('resize', resizeCanvas);
      if (animationFrame) cancelAnimationFrame(animationFrame);
    };
  }, [isProcessing, activeStep]);

  const handleDataSourceConnect = async (source) => {
    setDataSources(prev => ({
      ...prev,
      [source]: { ...prev[source], status: 'connecting' }
    }));

    // Simulate API connection
    setTimeout(() => {
      setDataSources(prev => ({
        ...prev,
        [source]: { connected: true, status: 'connected' }
      }));
      setActiveStep(2);
    }, 2000);
  };

  const handleFileUpload = (event) => {
    const files = Array.from(event.target.files);
    setDataSources(prev => ({
      ...prev,
      uploads: {
        ...prev.uploads,
        files: [...prev.uploads.files, ...files]
      }
    }));
  };

  const generateUserStories = async () => {
    setIsProcessing(true);
    setActiveStep(3);
    
    // Simulate processing steps
    setTimeout(() => setActiveStep(4), 3000);
    setTimeout(() => {
      setGeneratedStories([
        {
          id: 1,
          title: "User Authentication",
          description: "As a user, I want to securely log into the system so that I can access my personalized dashboard.",
          acceptanceCriteria: ["Valid credentials allow access", "Invalid credentials show error", "Password reset functionality available"],
          priority: "High",
          storyPoints: 5
        },
        {
          id: 2,
          title: "Data Visualization",
          description: "As an analyst, I want to view interactive charts so that I can better understand data trends.",
          acceptanceCriteria: ["Charts load within 3 seconds", "Interactive tooltips show details", "Export functionality available"],
          priority: "Medium",
          storyPoints: 8
        }
      ]);
      setIsProcessing(false);
    }, 5000);
  };

  const exportToJira = async () => {
    // Simulate Jira export
    alert('User stories exported to Jira successfully!');
  };

  const DataSourceCard = ({ title, icon: Icon, source, connected, status }) => (
    <div className={`p-6 rounded-lg border-2 transition-all duration-300 ${
      connected 
        ? 'border-green-400 bg-green-50' 
        : 'border-slate-300 bg-white hover:border-indigo-400'
    }`}>
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <Icon className={`w-8 h-8 ${connected ? 'text-green-600' : 'text-slate-600'}`} />
          <h3 className="text-lg font-semibold text-slate-800">{title}</h3>
        </div>
        {status === 'connecting' && <Loader2 className="w-5 h-5 animate-spin text-indigo-600" />}
        {connected && <CheckCircle className="w-5 h-5 text-green-600" />}
      </div>
      
      {source === 'uploads' ? (
        <div>
          <input
            type="file"
            multiple
            accept=".pdf,.doc,.docx,.txt"
            onChange={handleFileUpload}
            className="hidden"
            id="file-upload"
          />
          <label
            htmlFor="file-upload"
            className="cursor-pointer inline-flex items-center px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700"
          >
            <Upload className="w-4 h-4 mr-2" />
            Upload Files
          </label>
          {dataSources.uploads.files.length > 0 && (
            <div className="mt-4">
              <p className="text-sm text-slate-600 mb-2">Uploaded files:</p>
              {dataSources.uploads.files.map((file, idx) => (
                <div key={idx} className="flex items-center text-sm text-slate-700 mb-1">
                  <FileText className="w-4 h-4 mr-2" />
                  {file.name}
                </div>
              ))}
            </div>
          )}
        </div>
      ) : (
        !connected && (
          <button
            onClick={() => handleDataSourceConnect(source)}
            disabled={status === 'connecting'}
            className="w-full px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50"
          >
            {status === 'connecting' ? 'Connecting...' : 'Connect'}
          </button>
        )
      )}
    </div>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-indigo-50 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-slate-800 mb-4">
            RAG User Stories Generator
          </h1>
          <p className="text-xl text-slate-600">
            Transform requirements into actionable user stories using AI
          </p>
        </div>

        {/* Progress Steps */}
        <div className="flex justify-center mb-8">
          <div className="flex items-center space-x-4">
            {[1, 2, 3, 4].map((step) => (
              <div key={step} className="flex items-center">
                <div className={`w-10 h-10 rounded-full flex items-center justify-center border-2 ${
                  activeStep >= step 
                    ? 'bg-indigo-600 border-indigo-600 text-white' 
                    : 'bg-white border-slate-300 text-slate-400'
                }`}>
                  {step}
                </div>
                {step < 4 && (
                  <ChevronRight className={`w-5 h-5 mx-2 ${
                    activeStep > step ? 'text-indigo-600' : 'text-slate-300'
                  }`} />
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Main Content Area */}
        <div className="relative">
          {/* Animated Canvas */}
          <canvas
            ref={canvasRef}
            className="absolute inset-0 pointer-events-none z-0"
            style={{ width: '100%', height: '600px' }}
          />

          <div className="relative z-10 grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* Data Sources */}
            <div className="space-y-6">
              <h2 className="text-2xl font-bold text-slate-800 mb-4">Data Sources</h2>
              
              <DataSourceCard
                title="Jira"
                icon={Database}
                source="jira"
                connected={dataSources.jira.connected}
                status={dataSources.jira.status}
              />
              
              <DataSourceCard
                title="Confluence"
                icon={FileText}
                source="confluence"
                connected={dataSources.confluence.connected}
                status={dataSources.confluence.status}
              />
              
              <DataSourceCard
                title="SharePoint"
                icon={Database}
                source="sharepoint"
                connected={dataSources.sharepoint.connected}
                status={dataSources.sharepoint.status}
              />
              
              <DataSourceCard
                title="Upload Files"
                icon={Upload}
                source="uploads"
                connected={dataSources.uploads.files.length > 0}
                status={dataSources.uploads.status}
              />
            </div>

            {/* Processing Engine */}
            <div className="space-y-6">
              <h2 className="text-2xl font-bold text-slate-800 mb-4">AI Processing</h2>
              
              {/* LLM Configuration */}
              <div className="p-6 bg-white rounded-lg border-2 border-slate-300">
                <div className="flex items-center space-x-3 mb-4">
                  <Zap className="w-8 h-8 text-indigo-600" />
                  <h3 className="text-lg font-semibold text-slate-800">LLM Configuration</h3>
                </div>
                
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-2">
                      Provider
                    </label>
                    <select
                      value={llmConfig.provider}
                      onChange={(e) => setLlmConfig(prev => ({ ...prev, provider: e.target.value }))}
                      className="w-full p-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-indigo-500"
                    >
                      <option value="openai">OpenAI</option>
                      <option value="azure">Azure OpenAI</option>
                      <option value="gemini">Google Gemini</option>
                      <option value="claude">Anthropic Claude</option>
                    </select>
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-2">
                      Model
                    </label>
                    <select
                      value={llmConfig.model}
                      onChange={(e) => setLlmConfig(prev => ({ ...prev, model: e.target.value }))}
                      className="w-full p-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-indigo-500"
                    >
                      <option value="gpt-4">GPT-4</option>
                      <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
                      <option value="gemini-pro">Gemini Pro</option>
                      <option value="claude-3-sonnet">Claude 3 Sonnet</option>
                    </select>
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-2">
                      Temperature: {llmConfig.temperature}
                    </label>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.1"
                      value={llmConfig.temperature}
                      onChange={(e) => setLlmConfig(prev => ({ ...prev, temperature: parseFloat(e.target.value) }))}
                      className="w-full"
                    />
                  </div>
                </div>
              </div>

              {/* Generate Button */}
              <button
                onClick={generateUserStories}
                disabled={isProcessing || !Object.values(dataSources).some(ds => ds.connected || ds.files?.length > 0)}
                className="w-full px-6 py-4 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-lg font-semibold hover:from-indigo-700 hover:to-purple-700 disabled:opacity-50 flex items-center justify-center space-x-2"
              >
                {isProcessing ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    <span>Processing...</span>
                  </>
                ) : (
                  <>
                    <Play className="w-5 h-5" />
                    <span>Generate User Stories</span>
                  </>
                )}
              </button>
            </div>

            {/* Output */}
            <div className="space-y-6">
              <h2 className="text-2xl font-bold text-slate-800 mb-4">Generated Stories</h2>
              
              {generatedStories.length > 0 ? (
                <div className="space-y-4">
                  {generatedStories.map((story) => (
                    <div key={story.id} className="p-6 bg-white rounded-lg border-2 border-slate-300">
                      <div className="flex justify-between items-start mb-3">
                        <h3 className="text-lg font-semibold text-slate-800">{story.title}</h3>
                        <span className={`px-2 py-1 rounded text-xs font-medium ${
                          story.priority === 'High' ? 'bg-red-100 text-red-800' :
                          story.priority === 'Medium' ? 'bg-yellow-100 text-yellow-800' :
                          'bg-green-100 text-green-800'
                        }`}>
                          {story.priority}
                        </span>
                      </div>
                      
                      <p className="text-slate-700 mb-3">{story.description}</p>
                      
                      <div className="mb-3">
                        <h4 className="text-sm font-medium text-slate-800 mb-2">Acceptance Criteria:</h4>
                        <ul className="text-sm text-slate-600 space-y-1">
                          {story.acceptanceCriteria.map((criteria, idx) => (
                            <li key={idx} className="flex items-center">
                              <CheckCircle className="w-4 h-4 mr-2 text-green-500" />
                              {criteria}
                            </li>
                          ))}
                        </ul>
                      </div>
                      
                      <div className="text-sm text-slate-600">
                        Story Points: <span className="font-medium">{story.storyPoints}</span>
                      </div>
                    </div>
                  ))}
                  
                  <button
                    onClick={exportToJira}
                    className="w-full px-4 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 flex items-center justify-center space-x-2"
                  >
                    <Download className="w-5 h-5" />
                    <span>Export to Jira</span>
                  </button>
                </div>
              ) : (
                <div className="p-8 bg-white rounded-lg border-2 border-dashed border-slate-300 text-center">
                  <FileText className="w-12 h-12 text-slate-400 mx-auto mb-4" />
                  <p className="text-slate-600">Generated user stories will appear here</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RAGUserStoriesApp;