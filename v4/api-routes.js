// routes/auth.js - Authentication Routes
const express = require('express');
const joi = require('joi');
const User = require('../models/User');
const { authenticate } = require('../middleware/auth');
const router = express.Router();

// Validation schemas
const registerSchema = joi.object({
  email: joi.string().email().required(),
  password: joi.string().min(8).required(),
  firstName: joi.string().min(2).required(),
  lastName: joi.string().min(2).required()
});

const loginSchema = joi.object({
  email: joi.string().email().required(),
  password: joi.string().required()
});

// Register
router.post('/register', async (req, res) => {
  try {
    const { error } = registerSchema.validate(req.body);
    if (error) {
      return res.status(400).json({ error: error.details[0].message });
    }

    const { email, password, firstName, lastName } = req.body;

    // Check if user exists
    const existingUser = await User.findOne({ email });
    if (existingUser) {
      return res.status(400).json({ error: 'User already exists' });
    }

    // Create user
    const user = new User({
      email,
      password,
      firstName,
      lastName
    });

    await user.save();

    // Generate token
    const token = user.generateAuthToken();

    res.status(201).json({
      message: 'User created successfully',
      token,
      user: {
        id: user._id,
        email: user.email,
        firstName: user.firstName,
        lastName: user.lastName,
        role: user.role
      }
    });
  } catch (error) {
    console.error('Registration error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Login
router.post('/login', async (req, res) => {
  try {
    const { error } = loginSchema.validate(req.body);
    if (error) {
      return res.status(400).json({ error: error.details[0].message });
    }

    const { email, password } = req.body;

    // Find user
    const user = await User.findOne({ email });
    if (!user || !user.isActive) {
      return res.status(401).json({ error: 'Invalid credentials' });
    }

    // Check password
    const isMatch = await user.comparePassword(password);
    if (!isMatch) {
      return res.status(401).json({ error: 'Invalid credentials' });
    }

    // Generate token
    const token = user.generateAuthToken();

    res.json({
      message: 'Login successful',
      token,
      user: {
        id: user._id,
        email: user.email,
        firstName: user.firstName,
        lastName: user.lastName,
        role: user.role,
        preferences: user.preferences
      }
    });
  } catch (error) {
    console.error('Login error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Get current user
router.get('/me', authenticate, async (req, res) => {
  res.json({
    user: {
      id: req.user._id,
      email: req.user.email,
      firstName: req.user.firstName,
      lastName: req.user.lastName,
      role: req.user.role,
      preferences: req.user.preferences,
      integrations: req.user.integrations,
      usage: req.user.usage
    }
  });
});

// Update user preferences
router.patch('/preferences', authenticate, async (req, res) => {
  try {
    const allowedUpdates = ['defaultLLMProvider', 'defaultModel', 'temperature'];
    const updates = Object.keys(req.body);
    const isValidUpdate = updates.every(update => allowedUpdates.includes(update));

    if (!isValidUpdate) {
      return res.status(400).json({ error: 'Invalid updates' });
    }

    updates.forEach(update => {
      req.user.preferences[update] = req.body[update];
    });

    await req.user.save();
    res.json({ message: 'Preferences updated successfully', preferences: req.user.preferences });
  } catch (error) {
    res.status(500).json({ error: 'Internal server error' });
  }
});

module.exports = router;

---

// routes/documents.js - Document Management Routes
const express = require('express');
const multer = require('multer');
const { authenticate } = require('../middleware/auth');
const Document = require('../models/Document');
const DocumentProcessor = require('../services/DocumentProcessor');
const VectorService = require('../services/VectorService');
const router = express.Router();

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: 'uploads/',
  filename: (req, file, cb) => {
    const uniqueSuffix = `${Date.now()}-${Math.round(Math.random() * 1E9)}`;
    cb(null, `${uniqueSuffix}-${file.originalname}`);
  }
});

const upload = multer({
  storage,
  limits: {
    fileSize: 50 * 1024 * 1024, // 50MB
    files: 10
  },
  fileFilter: (req, file, cb) => {
    const allowedTypes = ['application/pdf', 'application/msword', 
                         'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                         'text/plain', 'text/markdown'];
    
    if (allowedTypes.includes(file.mimetype)) {
      cb(null, true);
    } else {
      cb(new Error('Invalid file type. Only PDF, Word, and text files are allowed.'));
    }
  }
});

// Upload files
router.post('/upload', authenticate, upload.array('files'), async (req, res) => {
  try {
    const processedDocuments = [];
    
    for (const file of req.files) {
      try {
        // Extract text content
        const content = await DocumentProcessor.extractTextFromFile(file.path, file.mimetype);
        
        // Create document record
        const document = new Document({
          userId: req.user._id,
          filename: file.filename,
          originalName: file.originalname,
          mimeType: file.mimetype,
          size: file.size,
          content,
          source: 'upload',
          status: 'processing'
        });

        // Generate chunks and embeddings
        const chunks = DocumentProcessor.chunkText(content);
        document.chunks = chunks.map((chunk, index) => ({
          index,
          content: chunk,
          embedding: [] // Will be populated by vector service
        }));

        await document.save();

        // Add to vector store
        await VectorService.addDocuments([{
          id: document._id,
          content,
          title: file.originalname,
          source: 'upload'
        }], req.user._id);

        document.status = 'completed';
        await document.save();

        processedDocuments.push({
          id: document._id,
          name: file.originalname,
          size: file.size,
          status: 'completed'
        });

      } catch (error) {
        console.error(`Error processing file ${file.originalname}:`, error);
        processedDocuments.push({
          name: file.originalname,
          status: 'failed',
          error: error.message
        });
      }
    }

    res.json({
      success: true,
      documents: processedDocuments,
      message: `Processed ${processedDocuments.length} files`
    });

  } catch (error) {
    console.error('Upload error:', error);
    res.status(500).json({ error: 'File upload failed' });
  }
});

// Get user documents
router.get('/', authenticate, async (req, res) => {
  try {
    const { page = 1, limit = 20, source, status } = req.query;
    
    const filter = { userId: req.user._id };
    if (source) filter.source = source;
    if (status) filter.status = status;

    const documents = await Document.find(filter)
      .select('-content -chunks') // Exclude large fields
      .sort({ createdAt: -1 })
      .limit(limit * 1)
      .skip((page - 1) * limit);

    const total = await Document.countDocuments(filter);

    res.json({
      documents,
      pagination: {
        page: parseInt(page),
        limit: parseInt(limit),
        total,
        pages: Math.ceil(total / limit)
      }
    });
  } catch (error) {
    res.status(500).json({ error: 'Failed to fetch documents' });
  }
});

// Get document content
router.get('/:id', authenticate, async (req, res) => {
  try {
    const document = await Document.findOne({
      _id: req.params.id,
      userId: req.user._id
    });

    if (!document) {
      return res.status(404).json({ error: 'Document not found' });
    }

    res.json(document);
  } catch (error) {
    res.status(500).json({ error: 'Failed to fetch document' });
  }
});

// Delete document
router.delete('/:id', authenticate, async (req, res) => {
  try {
    const document = await Document.findOneAndDelete({
      _id: req.params.id,
      userId: req.user._id
    });

    if (!document) {
      return res.status(404).json({ error: 'Document not found' });
    }

    // TODO: Remove from vector store
    // TODO: Delete physical file

    res.json({ message: 'Document deleted successfully' });
  } catch (error) {
    res.status(500).json({ error: 'Failed to delete document' });
  }
});

module.exports = router;

---

// routes/userStories.js - User Stories Routes
const express = require('express');
const joi = require('joi');
const { authenticate } = require('../middleware/auth');
const UserStory = require('../models/UserStory');
const LLMService = require('../services/LLMService');
const VectorService = require('../services/VectorService');
const DataSourceManager = require('../services/DataSourceManager');
const router = express.Router();

// Validation schema
const generateStoriesSchema = joi.object({
  requirements: joi.array().items(joi.string()).required(),
  llmConfig: joi.object({
    provider: joi.string().valid('openai', 'azure', 'gemini', 'claude').required(),
    model: joi.string().required(),
    temperature: joi.number().min(0).max(1).default(0.7),
    maxTokens: joi.number().min(100).max(8000).default(4000)
  }).required(),
  dataSources: joi.object({
    jira: joi.object({
      projectKey: joi.string()
    }),
    confluence: joi.object({
      spaceKey: joi.string()
    }),
    sharepoint: joi.object({
      driveId: joi.string()
    })
  }).default({})
});

// Generate user stories
router.post('/generate', authenticate, async (req, res) => {
  try {
    const { error } = generateStoriesSchema.validate(req.body);
    if (error) {
      return res.status(400).json({ error: error.details[0].message });
    }

    const { requirements, llmConfig, dataSources = {} } = req.body;

    // Get relevant context from vector store
    const queryText = requirements.join(' ');
    const relevantDocs = await VectorService.search(queryText, req.user._id, 10);
    const context = relevantDocs.map(doc => doc.content).join('\n\n');

    // Collect additional requirements from data sources
    let allRequirements = [...requirements];

    // Fetch from Jira if configured
    if (dataSources.jira?.projectKey && req.user.integrations.jira.connected) {
      try {
        const dataSourceManager = new DataSourceManager();
        await dataSourceManager.connectJira(req.user.integrations.jira);
        const jiraRequirements = await dataSourceManager.fetchJiraRequirements(dataSources.jira.projectKey);
        allRequirements = allRequirements.concat(
          jiraRequirements.map(req => `${req.summary}: ${req.description}`)
        );
      } catch (error) {
        console.error('Failed to fetch Jira requirements:', error);
      }
    }

    // Generate user stories
    const userStories = await LLMService.generateUserStories(
      context,
      allRequirements,
      llmConfig,
      req.user._id
    );

    // Save generated stories
    const savedStories = [];
    for (const story of userStories) {
      const userStory = new UserStory({
        userId: req.user._id,
        projectId: dataSources.jira?.projectKey || 'default',
        ...story,
        generationMetadata: {
          ...story.generationMetadata,
          sourceDocuments: relevantDocs.map(doc => doc.title || doc.source)
        }
      });

      await userStory.save();
      savedStories.push(userStory);
    }

    res.json({
      success: true,
      userStories: savedStories,
      metadata: {
        requirementsCount: allRequirements.length,
        documentsUsed: relevantDocs.length,
        llmProvider: llmConfig.provider,
        model: llmConfig.model,
        tokensUsed: userStories.reduce((sum, story) => 
          sum + (story.generationMetadata?.tokensUsed || 0), 0)
      }
    });

  } catch (error) {
    console.error('Error generating user stories:', error);
    res.status(500).json({ error: error.message });
  }
});

// Get user stories
router.get('/', authenticate, async (req, res) => {
  try {
    const { 
      page = 1, 
      limit = 20, 
      projectId, 
      status, 
      priority,
      sortBy = 'createdAt',
      sortOrder = 'desc'
    } = req.query;

    const filter = { userId: req.user._id };
    if (projectId) filter.projectId = projectId;
    if (status) filter.status = status;
    if (priority) filter.priority = priority;

    const sort = {};
    sort[sortBy] = sortOrder === 'desc' ? -1 : 1;

    const userStories = await UserStory.find(filter)
      .sort(sort)
      .limit(limit * 1)
      .skip((page - 1) * limit);

    const total = await UserStory.countDocuments(filter);

    res.json({
      userStories,
      pagination: {
        page: parseInt(page),
        limit: parseInt(limit),
        total,
        pages: Math.ceil(total / limit)
      }
    });
  } catch (error) {
    res.status(500).json({ error: 'Failed to fetch user stories' });
  }
});

// Get single user story
router.get('/:id', authenticate, async (req, res) => {
  try {
    const userStory = await UserStory.findOne({
      _id: req.params.id,
      userId: req.user._id
    });

    if (!userStory) {
      return res.status(404).json({ error: 'User story not found' });
    }

    res.json(userStory);
  } catch (error) {
    res.status(500).json({ error: 'Failed to fetch user story' });
  }
});

// Update user story
router.patch('/:id', authenticate, async (req, res) => {
  try {
    const allowedUpdates = [
      'title', 'description', 'acceptanceCriteria', 'priority', 
      'storyPoints', 'epic', 'labels', 'components'
    ];
    
    const updates = Object.keys(req.body);
    const isValidUpdate = updates.every(update => allowedUpdates.includes(update));

    if (!isValidUpdate) {
      return res.status(400).json({ error: 'Invalid updates' });
    }

    const userStory = await UserStory.findOneAndUpdate(
      { _id: req.params.id, userId: req.user._id },
      req.body,
      { new: true, runValidators: true }
    );

    if (!userStory) {
      return res.status(404).json({ error: 'User story not found' });
    }

    res.json(userStory);
  } catch (error) {
    res.status(500).json({ error: 'Failed to update user story' });
  }
});

// Delete user story
router.delete('/:id', authenticate, async (req, res) => {
  try {
    const userStory = await UserStory.findOneAndDelete({
      _id: req.params.id,
      userId: req.user._id
    });

    if (!userStory) {
      return res.status(404).json({ error: 'User story not found' });
    }

    res.json({ message: 'User story deleted successfully' });
  } catch (error) {
    res.status(500).json({ error: 'Failed to delete user story' });
  }
});

// Export to Jira
router.post('/export/jira', authenticate, async (req, res) => {
  try {
    const { userStoryIds, projectKey, issueTypeId = '7' } = req.body;

    if (!req.user.integrations.jira.connected) {
      return res.status(400).json({ error: 'Jira not connected' });
    }

    // Get user stories
    const userStories = await UserStory.find({
      _id: { $in: userStoryIds },
      userId: req.user._id
    });

    if (userStories.length === 0) {
      return res.status(404).json({ error: 'No user stories found' });
    }

    // Export to Jira
    const dataSourceManager = new DataSourceManager();
    await dataSourceManager.connectJira(req.user.integrations.jira);
    
    const createdIssues = await dataSourceManager.exportToJira(
      userStories,
      projectKey,
      issueTypeId
    );

    // Update user stories with Jira issue keys
    for (let i = 0; i < userStories.length; i++) {
      userStories[i].jiraIssueKey = createdIssues[i].key;
      userStories[i].status = 'exported';
      await userStories[i].save();
    }

    res.json({
      success: true,
      createdIssues,
      message: `Exported ${createdIssues.length} user stories to Jira`
    });

  } catch (error) {
    console.error('Jira export error:', error);
    res.status(500).json({ error: 'Failed to export to Jira' });
  }
});

// Get analytics
router.get('/analytics/summary', authenticate, async (req, res) => {
  try {
    const { projectId, dateFrom, dateTo } = req.query;

    const filter = { userId: req.user._id };
    if (projectId) filter.projectId = projectId;
    if (dateFrom || dateTo) {
      filter.createdAt = {};
      if (dateFrom) filter.createdAt.$gte = new Date(dateFrom);
      if (dateTo) filter.createdAt.$lte = new Date(dateTo);
    }

    const totalStories = await UserStory.countDocuments(filter);
    
    const priorityBreakdown = await UserStory.aggregate([
      { $match: filter },
      { $group: { _id: '$priority', count: { $sum: 1 } } }
    ]);

    const statusBreakdown = await UserStory.aggregate([
      { $match: filter },
      { $group: { _id: '$status', count: { $sum: 1 } } }
    ]);

    const storyPointsTotal = await UserStory.aggregate([
      { $match: filter },
      { $group: { _id: null, total: { $sum: '$storyPoints' } } }
    ]);

    res.json({
      totalStories,
      priorityBreakdown: priorityBreakdown.reduce((acc, item) => {
        acc[item._id] = item.count;
        return acc;
      }, {}),
      statusBreakdown: statusBreakdown.reduce((acc, item) => {
        acc[item._id] = item.count;
        return acc;
      }, {}),
      totalStoryPoints: storyPointsTotal[0]?.total || 0
    });

  } catch (error) {
    res.status(500).json({ error: 'Failed to fetch analytics' });
  }
});

module.exports = router;

---

// routes/integrations.js - Data Source Integration Routes
const express = require('express');
const joi = require('joi');
const { authenticate } = require('../middleware/auth');
const User = require('../models/User');
const DataSourceManager = require('../services/DataSourceManager');
const router = express.Router();

// Validation schemas
const jiraSchema = joi.object({
  url: joi.string().uri().required(),
  username: joi.string().email().required(),
  apiToken: joi.string().required()
});

const confluenceSchema = joi.object({
  url: joi.string().uri().required(),
  username: joi.string().email().required(),
  apiToken: joi.string().required()
});

const sharepointSchema = joi.object({
  tenantId: joi.string().required(),
  clientId: joi.string().required(),
  clientSecret: joi.string().required(),
  siteUrl: joi.string().uri().required()
});

// Connect Jira
router.post('/jira/connect', authenticate, async (req, res) => {
  try {
    const { error } = jiraSchema.validate(req.body);
    if (error) {
      return res.status(400).json({ error: error.details[0].message });
    }

    const dataSourceManager = new DataSourceManager();
    const result = await dataSourceManager.connectJira(req.body);

    // Save integration details
    req.user.integrations.jira = {
      ...req.body,
      connected: true
    };
    await req.user.save();

    res.json({
      success: true,
      message: 'Jira connected successfully',
      user: result.user
    });

  } catch (error) {
    console.error('Jira connection error:', error);
    res.status(400).json({ error: error.message });
  }
});

// Test Jira connection
router.get('/jira/test', authenticate, async (req, res) => {
  try {
    if (!req.user.integrations.jira.connected) {
      return res.status(400).json({ error: 'Jira not connected' });
    }

    const dataSourceManager = new DataSourceManager();
    await dataSourceManager.connectJira(req.user.integrations.jira);

    res.json({ success: true, message: 'Jira connection is active' });
  } catch (error) {
    res.status(400).json({ error: 'Jira connection failed', details: error.message });
  }
});

// Get Jira projects
router.get('/jira/projects', authenticate, async (req, res) => {
  try {
    if (!req.user.integrations.jira.connected) {
      return res.status(400).json({ error: 'Jira not connected' });
    }

    const dataSourceManager = new DataSourceManager();
    await dataSourceManager.connectJira(req.user.integrations.jira);
    
    const projects = await dataSourceManager.fetchJiraProjects();
    res.json(projects);
  } catch (error) {
    res.status(400).json({ error: 'Failed to fetch Jira projects' });
  }
});

// Connect Confluence
router.post('/confluence/connect', authenticate, async (req, res) => {
  try {
    const { error } = confluenceSchema.validate(req.body);
    if (error) {
      return res.status(400).json({ error: error.details[0].message });
    }

    const dataSourceManager = new DataSourceManager();
    const result = await dataSourceManager.connectConfluence(req.body);

    req.user.integrations.confluence = {
      ...req.body,
      connected: true
    };
    await req.user.save();

    res.json({
      success: true,
      message: 'Confluence connected successfully',
      user: result.user
    });

  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

// Get Confluence spaces
router.get('/confluence/spaces', authenticate, async (req, res) => {
  try {
    if (!req.user.integrations.confluence.connected) {
      return res.status(400).json({ error: 'Confluence not connected' });
    }

    const dataSourceManager = new DataSourceManager();
    await dataSourceManager.connectConfluence(req.user.integrations.confluence);
    
    const spaces = await dataSourceManager.fetchConfluenceSpaces();
    res.json(spaces);
  } catch (error) {
    res.status(400).json({ error: 'Failed to fetch Confluence spaces' });
  }
});

// Connect SharePoint
router.post('/sharepoint/connect', authenticate, async (req, res) => {
  try {
    const { error } = sharepointSchema.validate(req.body);
    if (error) {
      return res.status(400).json({ error: error.details[0].message });
    }

    const dataSourceManager = new DataSourceManager();
    const result = await dataSourceManager.connectSharePoint(req.body);

    req.user.integrations.sharepoint = {
      ...req.body,
      connected: true
    };
    await req.user.save();

    res.json({
      success: true,
      message: 'SharePoint connected successfully'
    });

  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

// Disconnect integration
router.delete('/:service/disconnect', authenticate, async (req, res) => {
  try {
    const { service } = req.params;
    
    if (!['jira', 'confluence', 'sharepoint'].includes(service)) {
      return res.status(400).json({ error: 'Invalid service' });
    }

    req.user.integrations[service] = {
      connected: false
    };
    await req.user.save();

    res.json({
      success: true,
      message: `${service} disconnected successfully`
    });

  } catch (error) {
    res.status(500).json({ error: 'Failed to disconnect service' });
  }
});

// Get integration status
router.get('/status', authenticate, async (req, res) => {
  try {
    const integrations = {
      jira: {
        connected: req.user.integrations.jira?.connected || false,
        url: req.user.integrations.jira?.url || null
      },
      confluence: {
        connected: req.user.integrations.confluence?.connected || false,
        url: req.user.integrations.confluence?.url || null
      },
      sharepoint: {
        connected: req.user.integrations.sharepoint?.connected || false,
        siteUrl: req.user.integrations.sharepoint?.siteUrl || null
      }
    };

    res.json(integrations);
  } catch (error) {
    res.status(500).json({ error: 'Failed to fetch integration status' });
  }
});

module.exports = router;

---

// app.js - Main Application Setup
const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const helmet = require('helmet');
const compression = require('compression');
const rateLimit = require('express-rate-limit');
const morgan = require('morgan');
const winston = require('winston');
require('dotenv').config();

// Import routes
const authRoutes = require('./routes/auth');
const documentRoutes = require('./routes/documents');
const userStoryRoutes = require('./routes/userStories');
const integrationRoutes = require('./routes/integrations');

// Import services
const VectorService = require('./services/VectorService');

const app = express();

// Configure Winston logger
const logger = winston.createLogger({
  level: process.env.LOG_LEVEL || 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.errors({ stack: true }),
    winston.format.json()
  ),
  transports: [
    new winston.transports.File({ filename: 'logs/error.log', level: 'error' }),
    new winston.transports.File({ filename: 'logs/combined.log' }),
    new winston.transports.Console({
      format: winston.format.simple()
    })
  ]
});

// Security middleware
app.use(helmet());

// CORS configuration
app.use(cors({
  origin: process.env.CORS_ORIGIN?.split(',') || 'http://localhost:3000',
  credentials: true
}));

// Rate limiting
const limiter = rateLimit({
  windowMs: parseInt(process.env.RATE_LIMIT_WINDOW_MS) || 15 * 60 * 1000,
  max: parseInt(process.env.RATE_LIMIT_MAX_REQUESTS) || 100,
  message: 'Too many requests from this IP, please try again later.'
});

app.use('/api/', limiter);

// Special rate limit for file uploads
const uploadLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 10,
  message: 'Too many uploads, please try again later.'
});

app.use('/api/documents/upload', uploadLimiter);

// Middleware
app.use(compression());
app.use(morgan('combined', { stream: { write: message => logger.info(message.trim()) } }));
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Health check
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
    environment: process.env.NODE_ENV
  });
});

// API routes
app.use('/api/auth', authRoutes);
app.use('/api/documents', documentRoutes);
app.use('/api/user-stories', userStoryRoutes);
app.use('/api/integrations', integrationRoutes);

// Global error handler
app.use((error, req, res, next) => {
  logger.error('Unhandled error:', error);
  
  if (error.name === 'ValidationError') {
    return res.status(400).json({
      error: 'Validation Error',
      details: Object.values(error.errors).map(e => e.message)
    });
  }
  
  if (error.name === 'CastError') {
    return res.status(400).json({ error: 'Invalid ID format' });
  }
  
  if (error.code === 11000) {
    return res.status(400).json({ error: 'Duplicate field value' });
  }

  res.status(500).json({
    error: 'Internal server error',
    message: process.env.NODE_ENV === 'development' ? error.message : 'Something went wrong'
  });
});

// 404 handler
app.use('*', (req, res) => {
  res.status(404).json({ error: 'Route not found' });
});

// Database connection and server startup
const startServer = async () => {
  try {
    // Connect to MongoDB
    await mongoose.connect(process.env.MONGODB_URI, {
      useNewUrlParser: true,
      useUnifiedTopology: true
    });
    logger.info('Connected to MongoDB');

    // Initialize vector service
    await VectorService.initialize();
    logger.info('Vector service initialized');

    // Start server
    const PORT = process.env.PORT || 3001;
    app.listen(PORT, () => {
      logger.info(`RAG User Stories API server running on port ${PORT}`);
      logger.info(`Environment: ${process.env.NODE_ENV}`);
      logger.info(`Health check: http://localhost:${PORT}/health`);
    });

  } catch (error) {
    logger.error('Failed to start server:', error);
    process.exit(1);
  }
};

// Graceful shutdown
process.on('SIGTERM', async () => {
  logger.info('SIGTERM received, shutting down gracefully');
  await mongoose.connection.close();
  process.exit(0);
});

process.on('SIGINT', async () => {
  logger.info('SIGINT received, shutting down gracefully');
  await mongoose.connection.close();
  process.exit(0);
});

if (require.main === module) {
  startServer();
}

module.exports = app;