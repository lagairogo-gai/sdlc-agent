import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Brain, 
  FileText, 
  Users, 
  Settings, 
  Plus,
  Zap,
  GitBranch,
  Database,
  Workflow,
  CheckCircle,
  AlertCircle,
  X
} from 'lucide-react';

// Toast Notification Component (replacing react-hot-toast)
const Toast = ({ message, type, onClose }) => {
  useEffect(() => {
    const timer = setTimeout(onClose, 3000);
    return () => clearTimeout(timer);
  }, [onClose]);

  return (
    <motion.div
      initial={{ opacity: 0, y: -50 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -50 }}
      className={`fixed top-4 right-4 z-50 p-4 rounded-lg border backdrop-blur-xl flex items-center space-x-3 ${
        type === 'success' 
          ? 'bg-green-500/20 border-green-400 text-green-400' 
          : 'bg-red-500/20 border-red-400 text-red-400'
      }`}
    >
      {type === 'success' ? <CheckCircle className="w-5 h-5" /> : <AlertCircle className="w-5 h-5" />}
      <span>{message}</span>
      <button onClick={onClose} className="ml-2">
        <X className="w-4 h-4" />
      </button>
    </motion.div>
  );
};

// Toast Manager
const useToast = () => {
  const [toasts, setToasts] = useState([]);

  const addToast = (message, type = 'success') => {
    const id = Date.now();
    setToasts(prev => [...prev, { id, message, type }]);
  };

  const removeToast = (id) => {
    setToasts(prev => prev.filter(toast => toast.id !== id));
  };

  const ToastContainer = () => (
    <div className="fixed top-0 right-0 z-50">
      <AnimatePresence>
        {toasts.map(toast => (
          <Toast
            key={toast.id}
            message={toast.message}
            type={toast.type}
            onClose={() => removeToast(toast.id)}
          />
        ))}
      </AnimatePresence>
    </div>
  );

  return { addToast, ToastContainer };
};

// Auth Store (using simple state management)
const useAuthStore = () => {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(() => {
    if (typeof window !== 'undefined') {
      return localStorage.getItem('token');
    }
    return null;
  });
  
  const login = (userData, authToken) => {
    setUser(userData);
    setToken(authToken);
    if (typeof window !== 'undefined') {
      localStorage.setItem('token', authToken);
      localStorage.setItem('user', JSON.stringify(userData));
    }
  };
  
  const logout = () => {
    setUser(null);
    setToken(null);
    if (typeof window !== 'undefined') {
      localStorage.removeItem('token');
      localStorage.removeItem('user');
    }
  };
  
  useEffect(() => {
    if (typeof window !== 'undefined') {
      const savedUser = localStorage.getItem('user');
      if (savedUser && token) {
        setUser(JSON.parse(savedUser));
      }
    }
  }, [token]);
  
  return { user, token, login, logout };
};

// Animated Background Component
const AnimatedBackground = () => {
  return (
    <div className="fixed inset-0 overflow-hidden pointer-events-none">
      <div className="absolute inset-0 bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
        {/* Animated particles */}
        {[...Array(20)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute w-2 h-2 bg-purple-400 rounded-full opacity-20"
            animate={{
              x: [0, 100, 0],
              y: [0, -100, 0],
              scale: [1, 1.5, 1],
            }}
            transition={{
              duration: 10 + i * 2,
              repeat: Infinity,
              ease: "easeInOut",
              delay: i * 0.5,
            }}
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
            }}
          />
        ))}
        
        {/* Gradient orbs */}
        <motion.div
          className="absolute w-96 h-96 bg-gradient-radial from-purple-500/20 to-transparent rounded-full blur-3xl"
          animate={{
            x: [0, 200, 0],
            y: [0, -200, 0],
            scale: [1, 1.2, 1],
          }}
          transition={{
            duration: 20,
            repeat: Infinity,
            ease: "easeInOut",
          }}
          style={{ left: '10%', top: '20%' }}
        />
        
        <motion.div
          className="absolute w-64 h-64 bg-gradient-radial from-blue-500/20 to-transparent rounded-full blur-3xl"
          animate={{
            x: [0, -150, 0],
            y: [0, 150, 0],
            scale: [1, 0.8, 1],
          }}
          transition={{
            duration: 15,
            repeat: Infinity,
            ease: "easeInOut",
            delay: 5,
          }}
          style={{ right: '15%', bottom: '25%' }}
        />
      </div>
    </div>
  );
};

// Sidebar Navigation
const Sidebar = ({ currentView, setCurrentView }) => {
  const menuItems = [
    { id: 'dashboard', icon: Brain, label: 'Dashboard', color: 'text-purple-400' },
    { id: 'projects', icon: Workflow, label: 'Projects', color: 'text-blue-400' },
    { id: 'stories', icon: FileText, label: 'User Stories', color: 'text-green-400' },
    { id: 'knowledge', icon: Database, label: 'Knowledge Graph', color: 'text-yellow-400' },
    { id: 'integrations', icon: GitBranch, label: 'Integrations', color: 'text-pink-400' },
    { id: 'settings', icon: Settings, label: 'Settings', color: 'text-gray-400' },
  ];

  return (
    <motion.div 
      className="w-64 bg-black/20 backdrop-blur-xl border-r border-white/10 flex flex-col"
      initial={{ x: -100, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      {/* Logo */}
      <div className="p-6 border-b border-white/10">
        <motion.div 
          className="flex items-center space-x-3"
          whileHover={{ scale: 1.05 }}
        >
          <div className="w-10 h-10 bg-gradient-to-br from-purple-500 to-blue-500 rounded-lg flex items-center justify-center">
            <Brain className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-white">StoryAI</h1>
            <p className="text-xs text-gray-400">Agent Platform</p>
          </div>
        </motion.div>
      </div>

      {/* Navigation */}
      <div className="flex-1 p-4 space-y-2">
        {menuItems.map((item) => (
          <motion.button
            key={item.id}
            onClick={() => setCurrentView(item.id)}
            className={`w-full flex items-center space-x-3 px-4 py-3 rounded-lg transition-all ${
              currentView === item.id
                ? 'bg-white/10 border border-white/20'
                : 'hover:bg-white/5'
            }`}
            whileHover={{ scale: 1.02, x: 4 }}
            whileTap={{ scale: 0.98 }}
          >
            <item.icon className={`w-5 h-5 ${item.color}`} />
            <span className="text-white font-medium">{item.label}</span>
            {currentView === item.id && (
              <motion.div
                className="ml-auto w-2 h-2 bg-purple-400 rounded-full"
                layoutId="activeIndicator"
              />
            )}
          </motion.button>
        ))}
      </div>

      {/* User Profile */}
      <div className="p-4 border-t border-white/10">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 bg-gradient-to-br from-green-400 to-blue-500 rounded-full flex items-center justify-center">
            <Users className="w-4 h-4 text-white" />
          </div>
          <div className="flex-1">
            <p className="text-sm font-medium text-white">John Doe</p>
            <p className="text-xs text-gray-400">Product Manager</p>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

// Connection Line Component (n8n style)
const ConnectionLine = ({ from, to, animated = true }) => {
  return (
    <motion.svg
      className="absolute inset-0 pointer-events-none"
      style={{ zIndex: 1 }}
    >
      <motion.path
        d={`M ${from.x} ${from.y} C ${from.x + 100} ${from.y} ${to.x - 100} ${to.y} ${to.x} ${to.y}`}
        stroke="url(#connectionGradient)"
        strokeWidth="2"
        fill="none"
        initial={{ pathLength: 0, opacity: 0 }}
        animate={{ pathLength: 1, opacity: animated ? 0.7 : 0.4 }}
        transition={{ duration: 1.5, ease: "easeInOut" }}
      />
      
      {animated && (
        <motion.circle
          r="4"
          fill="#8B5CF6"
          initial={{ opacity: 0 }}
          animate={{ 
            opacity: [0, 1, 0],
            offsetDistance: ["0%", "100%"]
          }}
          transition={{ 
            duration: 2,
            repeat: Infinity,
            ease: "easeInOut"
          }}
          style={{
            offsetPath: `path('M ${from.x} ${from.y} C ${from.x + 100} ${from.y} ${to.x - 100} ${to.y} ${to.x} ${to.y}')`
          }}
        />
      )}
      
      <defs>
        <linearGradient id="connectionGradient" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="#8B5CF6" stopOpacity="0.8" />
          <stop offset="50%" stopColor="#06B6D4" stopOpacity="0.8" />
          <stop offset="100%" stopColor="#10B981" stopOpacity="0.8" />
        </linearGradient>
      </defs>
    </motion.svg>
  );
};

// Dashboard Component
const Dashboard = () => {
  const [stats, setStats] = useState({
    totalProjects: 12,
    totalStories: 148,
    completedStories: 89,
    activeIntegrations: 3
  });

  const workflowNodes = [
    { id: 'requirements', label: 'Requirements', x: 100, y: 150, icon: FileText, color: 'from-blue-500 to-blue-600' },
    { id: 'rag', label: 'RAG System', x: 350, y: 100, icon: Database, color: 'from-purple-500 to-purple-600' },
    { id: 'kg', label: 'Knowledge Graph', x: 350, y: 200, icon: GitBranch, color: 'from-green-500 to-green-600' },
    { id: 'agent', label: 'AI Agent', x: 600, y: 150, icon: Brain, color: 'from-pink-500 to-pink-600' },
    { id: 'stories', label: 'User Stories', x: 850, y: 150, icon: Zap, color: 'from-yellow-500 to-yellow-600' },
  ];

  const connections = [
    { from: { x: 180, y: 150 }, to: { x: 350, y: 110 } },
    { from: { x: 180, y: 150 }, to: { x: 350, y: 190 } },
    { from: { x: 430, y: 120 }, to: { x: 600, y: 140 } },
    { from: { x: 430, y: 180 }, to: { x: 600, y: 160 } },
    { from: { x: 680, y: 150 }, to: { x: 850, y: 150 } },
  ];

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white">AI Agent Dashboard</h1>
          <p className="text-gray-400 mt-1">Orchestrate your user story generation pipeline</p>
        </div>
        <motion.button
          className="px-6 py-3 bg-gradient-to-r from-purple-500 to-blue-500 text-white rounded-lg font-medium flex items-center space-x-2 hover:shadow-lg hover:shadow-purple-500/25"
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          <Plus className="w-5 h-5" />
          <span>New Project</span>
        </motion.button>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        {[
          { label: 'Total Projects', value: stats.totalProjects, icon: Workflow, color: 'text-blue-400' },
          { label: 'User Stories', value: stats.totalStories, icon: FileText, color: 'text-green-400' },
          { label: 'Completed', value: stats.completedStories, icon: Zap, color: 'text-yellow-400' },
          { label: 'Integrations', value: stats.activeIntegrations, icon: GitBranch, color: 'text-pink-400' },
        ].map((stat, index) => (
          <motion.div
            key={stat.label}
            className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-xl p-6"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            whileHover={{ scale: 1.02, backgroundColor: 'rgba(255,255,255,0.08)' }}
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm">{stat.label}</p>
                <motion.p 
                  className="text-2xl font-bold text-white mt-1"
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ delay: index * 0.1 + 0.3, type: "spring" }}
                >
                  {stat.value}
                </motion.p>
              </div>
              <stat.icon className={`w-8 h-8 ${stat.color}`} />
            </div>
          </motion.div>
        ))}
      </div>

      {/* Workflow Visualization */}
      <motion.div
        className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-xl p-8"
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.5 }}
      >
        <h2 className="text-xl font-bold text-white mb-6">AI Agent Workflow</h2>
        
        <div className="relative h-96 overflow-hidden">
          {/* Connection Lines */}
          {connections.map((connection, index) => (
            <ConnectionLine
              key={index}
              from={connection.from}
              to={connection.to}
              animated={true}
            />
          ))}
          
          {/* Workflow Nodes */}
          {workflowNodes.map((node, index) => (
            <motion.div
              key={node.id}
              className={`absolute w-24 h-24 bg-gradient-to-br ${node.color} rounded-xl flex flex-col items-center justify-center text-white font-medium shadow-lg border border-white/20`}
              style={{ left: node.x, top: node.y, zIndex: 10 }}
              initial={{ opacity: 0, scale: 0 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: index * 0.2 + 0.8, type: "spring" }}
              whileHover={{ 
                scale: 1.1, 
                boxShadow: "0 20px 40px rgba(0,0,0,0.3)",
                zIndex: 20
              }}
            >
              <node.icon className="w-6 h-6 mb-1" />
              <span className="text-xs text-center px-2">{node.label}</span>
              
              {/* Pulse animation */}
              <motion.div
                className="absolute inset-0 rounded-xl border-2 border-white/50"
                animate={{ scale: [1, 1.2, 1], opacity: [0.5, 0, 0.5] }}
                transition={{ duration: 2, repeat: Infinity, delay: index * 0.3 }}
              />
            </motion.div>
          ))}
        </div>
        
        <div className="mt-6 text-center">
          <p className="text-gray-400">Real-time AI pipeline processing requirements into actionable user stories</p>
        </div>
      </motion.div>

      {/* Recent Activity */}
      <motion.div
        className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-xl p-6"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.8 }}
      >
        <h2 className="text-xl font-bold text-white mb-4">Recent Activity</h2>
        <div className="space-y-3">
          {[
            { action: 'Generated 5 user stories', project: 'E-commerce Platform', time: '2 minutes ago', status: 'success' },
            { action: 'Synced from Jira', project: 'Mobile App', time: '15 minutes ago', status: 'success' },
            { action: 'Updated knowledge graph', project: 'Analytics Dashboard', time: '1 hour ago', status: 'success' },
            { action: 'Export to Jira failed', project: 'CRM System', time: '2 hours ago', status: 'error' },
          ].map((activity, index) => (
            <motion.div
              key={index}
              className="flex items-center justify-between p-3 bg-white/3 rounded-lg border border-white/5"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.9 + index * 0.1 }}
            >
              <div className="flex items-center space-x-3">
                <div className={`w-2 h-2 rounded-full ${
                  activity.status === 'success' ? 'bg-green-400' : 'bg-red-400'
                }`} />
                <div>
                  <p className="text-white font-medium">{activity.action}</p>
                  <p className="text-gray-400 text-sm">{activity.project}</p>
                </div>
              </div>
              <span className="text-gray-400 text-sm">{activity.time}</span>
            </motion.div>
          ))}
        </div>
      </motion.div>
    </div>
  );
};

// Projects Component
const Projects = () => {
  const [projects, setProjects] = useState([
    {
      id: 1,
      name: 'E-commerce Platform',
      description: 'Complete online shopping solution with AI recommendations',
      status: 'active',
      storiesCount: 45,
      completedStories: 28,
      lastUpdated: '2 hours ago',
      integrations: ['jira', 'confluence']
    },
    {
      id: 2,
      name: 'Mobile Banking App',
      description: 'Secure mobile banking application with biometric authentication',
      status: 'active',
      storiesCount: 32,
      completedStories: 18,
      lastUpdated: '1 day ago',
      integrations: ['jira', 'sharepoint']
    },
    {
      id: 3,
      name: 'Analytics Dashboard',
      description: 'Real-time business intelligence and analytics platform',
      status: 'completed',
      storiesCount: 28,
      completedStories: 28,
      lastUpdated: '3 days ago',
      integrations: ['confluence']
    }
  ]);

  const { addToast } = useToast();

  const handleCreateProject = () => {
    addToast('New project creation coming soon!', 'success');
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold text-white">Projects</h1>
        <motion.button
          className="px-4 py-2 bg-gradient-to-r from-purple-500 to-blue-500 text-white rounded-lg flex items-center space-x-2"
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={handleCreateProject}
        >
          <Plus className="w-4 h-4" />
          <span>New Project</span>
        </motion.button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {projects.map((project, index) => (
          <motion.div
            key={project.id}
            className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-xl p-6 hover:bg-white/8 cursor-pointer"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            whileHover={{ scale: 1.02 }}
          >
            <div className="flex items-start justify-between mb-4">
              <h3 className="text-lg font-bold text-white">{project.name}</h3>
              <span className={`px-2 py-1 text-xs rounded-full ${
                project.status === 'active' 
                  ? 'bg-green-500/20 text-green-400' 
                  : 'bg-blue-500/20 text-blue-400'
              }`}>
                {project.status}
              </span>
            </div>
            
            <p className="text-gray-400 text-sm mb-4">{project.description}</p>
            
            <div className="space-y-3">
              <div className="flex justify-between text-sm">
                <span className="text-gray-400">Progress</span>
                <span className="text-white">
                  {project.completedStories}/{project.storiesCount} stories
                </span>
              </div>
              
              <div className="w-full bg-gray-700 rounded-full h-2">
                <motion.div
                  className="bg-gradient-to-r from-green-400 to-blue-500 h-2 rounded-full"
                  initial={{ width: 0 }}
                  animate={{ width: `${(project.completedStories / project.storiesCount) * 100}%` }}
                  transition={{ delay: index * 0.2 + 0.5, duration: 1 }}
                />
              </div>
              
              <div className="flex items-center justify-between">
                <div className="flex space-x-1">
                  {project.integrations.map((integration) => (
                    <div
                      key={integration}
                      className="w-6 h-6 bg-gradient-to-br from-purple-500 to-blue-500 rounded-full flex items-center justify-center"
                      title={integration}
                    >
                      <span className="text-white text-xs font-bold">
                        {integration[0].toUpperCase()}
                      </span>
                    </div>
                  ))}
                </div>
                <span className="text-gray-400 text-xs">{project.lastUpdated}</span>
              </div>
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );
};

// User Stories Component with Kanban Board
const UserStories = () => {
  const [stories] = useState({
    backlog: [
      { id: 1, title: 'User Authentication', description: 'As a user, I want to log in...', points: 5, priority: 'High' },
      { id: 2, title: 'Product Search', description: 'As a customer, I want to search...', points: 3, priority: 'Medium' },
    ],
    inProgress: [
      { id: 3, title: 'Shopping Cart', description: 'As a customer, I want to add items...', points: 8, priority: 'High' },
    ],
    review: [
      { id: 4, title: 'Payment Processing', description: 'As a customer, I want to pay...', points: 13, priority: 'Critical' },
    ],
    done: [
      { id: 5, title: 'User Profile', description: 'As a user, I want to manage...', points: 5, priority: 'Medium' },
    ]
  });

  const { addToast } = useToast();

  const columns = [
    { id: 'backlog', title: 'Backlog', color: 'from-gray-500 to-gray-600' },
    { id: 'inProgress', title: 'In Progress', color: 'from-blue-500 to-blue-600' },
    { id: 'review', title: 'Review', color: 'from-yellow-500 to-yellow-600' },
    { id: 'done', title: 'Done', color: 'from-green-500 to-green-600' },
  ];

  const getPriorityColor = (priority) => {
    switch (priority) {
      case 'Critical': return 'text-red-400 bg-red-500/20';
      case 'High': return 'text-orange-400 bg-orange-500/20';
      case 'Medium': return 'text-yellow-400 bg-yellow-500/20';
      case 'Low': return 'text-green-400 bg-green-500/20';
      default: return 'text-gray-400 bg-gray-500/20';
    }
  };

  const handleGenerateStories = () => {
    addToast('AI story generation feature coming soon!', 'success');
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold text-white">User Stories</h1>
        <motion.button
          className="px-4 py-2 bg-gradient-to-r from-purple-500 to-blue-500 text-white rounded-lg flex items-center space-x-2"
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={handleGenerateStories}
        >
          <Brain className="w-4 h-4" />
          <span>Generate Stories</span>
        </motion.button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        {columns.map((column, columnIndex) => (
          <motion.div
            key={column.id}
            className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-xl p-4"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: columnIndex * 0.1 }}
          >
            <div className="flex items-center space-x-2 mb-4">
              <div className={`w-3 h-3 rounded-full bg-gradient-to-r ${column.color}`} />
              <h3 className="font-bold text-white">{column.title}</h3>
              <span className="text-gray-400 text-sm">({stories[column.id]?.length || 0})</span>
            </div>
            
            <div className="space-y-3">
              {stories[column.id]?.map((story, index) => (
                <motion.div
                  key={story.id}
                  className="bg-white/5 border border-white/10 rounded-lg p-3 cursor-move"
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: columnIndex * 0.1 + index * 0.05 }}
                  whileHover={{ scale: 1.02, backgroundColor: 'rgba(255,255,255,0.08)' }}
                  drag
                  dragConstraints={{ left: 0, right: 0, top: 0, bottom: 0 }}
                >
                  <div className="flex items-start justify-between mb-2">
                    <h4 className="font-medium text-white text-sm">{story.title}</h4>
                    <span className={`px-2 py-1 text-xs rounded-full ${getPriorityColor(story.priority)}`}>
                      {story.priority}
                    </span>
                  </div>
                  <p className="text-gray-400 text-xs mb-3">{story.description}</p>
                  <div className="flex items-center justify-between">
                    <span className="text-purple-400 text-xs font-bold">{story.points} pts</span>
                    <div className="w-6 h-6 bg-gradient-to-br from-blue-500 to-purple-500 rounded-full flex items-center justify-center">
                      <FileText className="w-3 h-3 text-white" />
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );
};

// Main App Component
const App = () => {
  const [currentView, setCurrentView] = useState('dashboard');
  const { user, token, login, logout } = useAuthStore();
  const { addToast, ToastContainer } = useToast();

  const renderCurrentView = () => {
    switch (currentView) {
      case 'dashboard':
        return <Dashboard />;
      case 'projects':
        return <Projects />;
      case 'stories':
        return <UserStories />;
      case 'knowledge':
        return (
          <div className="text-white p-8">
            <h1 className="text-3xl font-bold mb-4">Knowledge Graph</h1>
            <div className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-xl p-8 text-center">
              <Database className="w-16 h-16 text-purple-400 mx-auto mb-4" />
              <p className="text-gray-400">Knowledge Graph visualization coming soon!</p>
              <p className="text-sm text-gray-500 mt-2">
                This will show entity relationships and project insights
              </p>
            </div>
          </div>
        );
      case 'integrations':
        return (
          <div className="text-white p-8">
            <h1 className="text-3xl font-bold mb-4">Integrations</h1>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {['Jira', 'Confluence', 'SharePoint'].map((integration, index) => (
                <motion.div
                  key={integration}
                  className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-xl p-6 text-center"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                  whileHover={{ scale: 1.05 }}
                >
                  <GitBranch className="w-12 h-12 text-blue-400 mx-auto mb-4" />
                  <h3 className="text-lg font-bold text-white mb-2">{integration}</h3>
                  <p className="text-gray-400 text-sm">Connect to {integration} to sync data</p>
                  <motion.button
                    className="mt-4 px-4 py-2 bg-blue-500/20 text-blue-400 rounded-lg border border-blue-400/30 hover:bg-blue-500/30 transition-colors"
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={() => addToast(`${integration} integration coming soon!`, 'success')}
                  >
                    Configure
                  </motion.button>
                </motion.div>
              ))}
            </div>
          </div>
        );
      case 'settings':
        return (
          <div className="text-white p-8">
            <h1 className="text-3xl font-bold mb-4">Settings</h1>
            <div className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-xl p-8">
              <div className="space-y-6">
                <div>
                  <h3 className="text-lg font-bold text-white mb-4">LLM Configuration</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {['OpenAI', 'Anthropic', 'Google', 'Azure'].map((provider) => (
                      <div key={provider} className="p-4 bg-white/5 rounded-lg border border-white/10">
                        <h4 className="font-medium text-white mb-2">{provider}</h4>
                        <input
                          type="password"
                          placeholder="API Key"
                          className="w-full p-2 bg-black/20 border border-white/20 rounded text-white placeholder-gray-400"
                        />
                      </div>
                    ))}
                  </div>
                </div>
                <div>
                  <h3 className="text-lg font-bold text-white mb-4">User Preferences</h3>
                  <div className="space-y-3">
                    <label className="flex items-center space-x-3">
                      <input type="checkbox" className="rounded" />
                      <span className="text-gray-400">Enable email notifications</span>
                    </label>
                    <label className="flex items-center space-x-3">
                      <input type="checkbox" className="rounded" defaultChecked />
                      <span className="text-gray-400">Auto-save generated stories</span>
                    </label>
                  </div>
                </div>
              </div>
            </div>
          </div>
        );
      default:
        return <Dashboard />;
    }
  };

  if (!token) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex items-center justify-center">
        <AnimatedBackground />
        <motion.div
          className="bg-white/10 backdrop-blur-xl border border-white/20 rounded-2xl p-8 w-96 relative z-10"
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5 }}
        >
          <div className="text-center mb-8">
            <motion.div
              className="w-16 h-16 bg-gradient-to-br from-purple-500 to-blue-500 rounded-xl flex items-center justify-center mx-auto mb-4"
              animate={{ rotate: [0, 360] }}
              transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
            >
              <Brain className="w-8 h-8 text-white" />
            </motion.div>
            <h1 className="text-2xl font-bold text-white">StoryAI Agent</h1>
            <p className="text-gray-400">Welcome back</p>
          </div>
          
          <motion.button
            className="w-full py-3 bg-gradient-to-r from-purple-500 to-blue-500 text-white rounded-lg font-medium"
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={() => {
              login({ name: 'John Doe', email: 'john@example.com' }, 'demo-token');
              addToast('Welcome to StoryAI Agent!', 'success');
            }}
          >
            Sign In
          </motion.button>
        </motion.div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex">
      <AnimatedBackground />
      
      <Sidebar currentView={currentView} setCurrentView={setCurrentView} />
      
      <main className="flex-1 overflow-auto relative z-10">
        <div className="p-8">
          <AnimatePresence mode="wait">
            <motion.div
              key={currentView}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.3 }}
            >
              {renderCurrentView()}
            </motion.div>
          </AnimatePresence>
        </div>
      </main>
      
      <ToastContainer />
    </div>
  );
};

export default App;