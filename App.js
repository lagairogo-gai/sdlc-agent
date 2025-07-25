// Update the frontend/src/App.js file with proper API calls and error handling

import React, { useState, useEffect } from 'react';
import { Play, GitBranch, Settings, Activity, AlertCircle, CheckCircle } from 'lucide-react';
import './App.css';

function App() {
  const [workflows, setWorkflows] = useState([]);
  const [agents, setAgents] = useState([
    { name: 'Requirements', status: 'connected' },
    { name: 'Design', status: 'connected' },
    { name: 'Code', status: 'connected' },
    { name: 'Quality', status: 'connected' },
    { name: 'Testing', status: 'connected' },
    { name: 'CI/CD', status: 'connected' },
    { name: 'Deployment', status: 'connected' },
    { name: 'Monitoring', status: 'connected' },
    { name: 'Maintenance', status: 'connected' }
  ]);

  const [newProject, setNewProject] = useState({
    name: '',
    description: '',
    requirements: ''
  });

  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [systemStatus, setSystemStatus] = useState('checking');

  // API base URL - this is crucial for proper communication
  const API_BASE_URL = process.env.NODE_ENV === 'production' 
    ? `http://${window.location.hostname}:8000`
    : 'http://localhost:8000';

  useEffect(() => {
    checkSystemHealth();
    fetchWorkflows();
    
    // Set up polling for workflow updates
    const interval = setInterval(() => {
      fetchWorkflows();
    }, 5000); // Poll every 5 seconds

    return () => clearInterval(interval);
  }, []);

  const checkSystemHealth = async () => {
    try {
      console.log('Checking system health at:', `${API_BASE_URL}/health`);
      const response = await fetch(`${API_BASE_URL}/health`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (response.ok) {
        const data = await response.json();
        console.log('System health:', data);
        setSystemStatus('healthy');
        setError('');
      } else {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
    } catch (error) {
      console.error('System health check failed:', error);
      setSystemStatus('unhealthy');
      setError(`Backend connection failed: ${error.message}`);
    }
  };

  const fetchWorkflows = async () => {
    try {
      console.log('Fetching workflows from:', `${API_BASE_URL}/workflows`);
      const response = await fetch(`${API_BASE_URL}/workflows`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (response.ok) {
        const data = await response.json();
        console.log('Workflows data:', data);
        setWorkflows(data.workflows || []);
      } else {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
    } catch (error) {
      console.error('Error fetching workflows:', error);
      // Don't show error if system is just starting up
      if (systemStatus === 'healthy') {
        setError(`Failed to fetch workflows: ${error.message}`);
      }
    }
  };

  const createWorkflow = async () => {
    if (!newProject.name.trim()) {
      setError('Project name is required');
      return;
    }

    setIsLoading(true);
    setError('');

    try {
      console.log('Creating workflow with data:', newProject);
      
      // Step 1: Create the workflow
      const createResponse = await fetch(`${API_BASE_URL}/workflows`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          project_id: `project_${Date.now()}`,
          name: newProject.name,
          description: newProject.description,
          requirements: { 
            description: newProject.requirements,
            details: newProject.requirements.split('\n').filter(line => line.trim())
          }
        }),
      });
      
      if (!createResponse.ok) {
        throw new Error(`Failed to create workflow: HTTP ${createResponse.status}`);
      }
      
      const createData = await createResponse.json();
      console.log('Workflow created:', createData);
      
      // Step 2: Execute the workflow
      const executeResponse = await fetch(`${API_BASE_URL}/workflows/${createData.workflow_id}/execute`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        }
      });
      
      if (!executeResponse.ok) {
        throw new Error(`Failed to execute workflow: HTTP ${executeResponse.status}`);
      }
      
      const executeData = await executeResponse.json();
      console.log('Workflow execution started:', executeData);
      
      // Reset form and refresh workflows
      setNewProject({ name: '', description: '', requirements: '' });
      fetchWorkflows();
      
      setError(''); // Clear any previous errors
      
    } catch (error) {
      console.error('Error creating/executing workflow:', error);
      setError(`Failed to create workflow: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  // Function to get status color
  const getStatusColor = (status) => {
    switch (status) {
      case 'completed': return 'bg-green-100 text-green-800';
      case 'running': return 'bg-blue-100 text-blue-800';
      case 'failed': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Header */}
      <header className="bg-white shadow-md">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div className="flex items-center">
              <GitBranch className="h-8 w-8 text-blue-600 mr-3" />
              <h1 className="text-2xl font-bold text-gray-900">
                Agentic AI SDLC System
              </h1>
            </div>
            <div className="flex items-center space-x-4">
              <div className={`flex items-center ${
                systemStatus === 'healthy' ? 'text-green-600' : 
                systemStatus === 'unhealthy' ? 'text-red-600' : 'text-yellow-600'
              }`}>
                {systemStatus === 'healthy' ? <CheckCircle className="h-5 w-5 mr-2" /> :
                 systemStatus === 'unhealthy' ? <AlertCircle className="h-5 w-5 mr-2" /> :
                 <Activity className="h-5 w-5 mr-2" />}
                <span className="text-sm font-medium">
                  {systemStatus === 'healthy' ? 'System Online' : 
                   systemStatus === 'unhealthy' ? 'System Offline' : 'Checking...'}
                </span>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Error Banner */}
      {error && (
        <div className="bg-red-50 border-l-4 border-red-400 p-4 mx-4 mt-4">
          <div className="flex">
            <AlertCircle className="h-5 w-5 text-red-400 mr-2" />
            <div>
              <p className="text-sm text-red-700">{error}</p>
              <p className="text-xs text-red-600 mt-1">
                API URL: {API_BASE_URL} | Check browser console for details
              </p>
            </div>
          </div>
        </div>
      )}

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          
          {/* Create New Project */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">
                Create New Project
              </h2>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Project Name *
                  </label>
                  <input
                    type="text"
                    value={newProject.name}
                    onChange={(e) => setNewProject({...newProject, name: e.target.value})}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    placeholder="My Awesome Project"
                    disabled={isLoading}
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Description
                  </label>
                  <textarea
                    value={newProject.description}
                    onChange={(e) => setNewProject({...newProject, description: e.target.value})}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    rows="3"
                    placeholder="Brief description of your project..."
                    disabled={isLoading}
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Requirements
                  </label>
                  <textarea
                    value={newProject.requirements}
                    onChange={(e) => setNewProject({...newProject, requirements: e.target.value})}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    rows="3"
                    placeholder="List your project requirements..."
                    disabled={isLoading}
                  />
                </div>
                
                <button
                  onClick={createWorkflow}
                  disabled={!newProject.name.trim() || isLoading || systemStatus !== 'healthy'}
                  className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white font-medium py-2 px-4 rounded-md transition duration-200 flex items-center justify-center"
                >
                  {isLoading ? (
                    <>
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                      Creating Workflow...
                    </>
                  ) : (
                    <>
                      <Play className="h-4 w-4 mr-2" />
                      Start SDLC Workflow
                    </>
                  )}
                </button>
                
                {systemStatus !== 'healthy' && (
                  <p className="text-xs text-red-600 text-center">
                    System must be online to create workflows
                  </p>
                )}
              </div>
            </div>
            
            {/* Agent Status */}
            <div className="bg-white rounded-lg shadow p-6 mt-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">
                Agent Status
              </h2>
              <div className="space-y-3">
                {agents.map((agent, index) => (
                  <div key={index} className="flex items-center justify-between">
                    <span className="text-sm font-medium text-gray-700">
                      {agent.name} Agent
                    </span>
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                      agent.status === 'connected' 
                        ? 'bg-green-100 text-green-800' 
                        : 'bg-red-100 text-red-800'
                    }`}>
                      {agent.status}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Active Workflows */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-lg font-semibold text-gray-900">
                  Active Workflows
                </h2>
                <button
                  onClick={fetchWorkflows}
                  className="text-blue-600 hover:text-blue-800 text-sm font-medium"
                  disabled={isLoading}
                >
                  Refresh
                </button>
              </div>
              
              {workflows.length === 0 ? (
                <div className="text-center py-8">
                  <Settings className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                  <p className="text-gray-500">No active workflows</p>
                  <p className="text-sm text-gray-400">Create a new project to get started</p>
                </div>
              ) : (
                <div className="space-y-4">
                  {workflows.map((workflow) => (
                    <div key={workflow.id} className="border border-gray-200 rounded-lg p-4">
                      <div className="flex items-center justify-between mb-2">
                        <h3 className="font-medium text-gray-900">{workflow.name}</h3>
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(workflow.status)}`}>
                          {workflow.status}
                        </span>
                      </div>
                      
                      <p className="text-sm text-gray-600 mb-3">{workflow.description}</p>
                      
                      {/* Progress Bar */}
                      <div className="space-y-2">
                        <div className="flex justify-between text-xs text-gray-500">
                          <span>Progress</span>
                          <span>
                            {workflow.phases ? workflow.phases.filter(p => p.status === 'completed').length : 0} / {workflow.phases ? workflow.phases.length : 9}
                          </span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2">
                          <div 
                            className="bg-blue-600 h-2 rounded-full transition-all duration-500"
                            style={{
                              width: workflow.phases ? `${(workflow.phases.filter(p => p.status === 'completed').length / workflow.phases.length) * 100}%` : '0%'
                            }}
                          ></div>
                        </div>
                      </div>
                      
                      {/* Phase Details */}
                      {workflow.phases && (
                        <div className="mt-3 grid grid-cols-3 gap-2">
                          {workflow.phases.map((phase, index) => (
                            <div key={index} className="text-center">
                              <div className={`w-8 h-8 rounded-full mx-auto mb-1 flex items-center justify-center text-xs font-medium ${
                                phase.status === 'completed'
                                  ? 'bg-green-500 text-white'
                                  : phase.status === 'running'
                                  ? 'bg-blue-500 text-white animate-pulse'
                                  : 'bg-gray-300 text-gray-600'
                              }`}>
                                {index + 1}
                              </div>
                              <div className="text-xs text-gray-600 capitalize">
                                {phase.name}
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                      
                      {/* Timestamps */}
                      <div className="mt-3 text-xs text-gray-500">
                        Created: {new Date(workflow.created_at).toLocaleString()}
                        {workflow.started_at && (
                          <span className="ml-4">
                            Started: {new Date(workflow.started_at).toLocaleString()}
                          </span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;