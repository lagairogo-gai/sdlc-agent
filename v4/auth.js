// middleware/auth.js - Authentication Middleware
const jwt = require('jsonwebtoken');
const User = require('../models/User');

const authenticate = async (req, res, next) => {
  try {
    const token = req.header('Authorization')?.replace('Bearer ', '');
    
    if (!token) {
      return res.status(401).json({ error: 'Access denied. No token provided.' });
    }

    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    const user = await User.findById(decoded.id).select('-password');
    
    if (!user || !user.isActive) {
      return res.status(401).json({ error: 'Invalid token or inactive user.' });
    }

    req.user = user;
    next();
  } catch (error) {
    res.status(401).json({ error: 'Invalid token.' });
  }
};

const authorize = (...roles) => {
  return (req, res, next) => {
    if (!roles.includes(req.user.role)) {
      return res.status(403).json({ error: 'Access denied. Insufficient permissions.' });
    }
    next();
  };
};

module.exports = { authenticate, authorize };

---

// models/User.js - User Model (Mongoose)
const mongoose = require('mongoose');
const bcrypt = require('bcryptjs');

const userSchema = new mongoose.Schema({
  email: {
    type: String,
    required: true,
    unique: true,
    lowercase: true,
    trim: true
  },
  password: {
    type: String,
    required: true,
    minlength: 8
  },
  firstName: {
    type: String,
    required: true,
    trim: true
  },
  lastName: {
    type: String,
    required: true,
    trim: true
  },
  role: {
    type: String,
    enum: ['admin', 'user', 'viewer'],
    default: 'user'
  },
  isActive: {
    type: Boolean,
    default: true
  },
  preferences: {
    defaultLLMProvider: {
      type: String,
      enum: ['openai', 'azure', 'gemini', 'claude'],
      default: 'openai'
    },
    defaultModel: String,
    temperature: {
      type: Number,
      min: 0,
      max: 1,
      default: 0.7
    }
  },
  integrations: {
    jira: {
      url: String,
      username: String,
      apiToken: String,
      connected: { type: Boolean, default: false }
    },
    confluence: {
      url: String,
      username: String,
      apiToken: String,
      connected: { type: Boolean, default: false }
    },
    sharepoint: {
      tenantId: String,
      clientId: String,
      clientSecret: String,
      connected: { type: Boolean, default: false }
    }
  },
  usage: {
    apiCalls: { type: Number, default: 0 },
    tokensUsed: { type: Number, default: 0 },
    storiesGenerated: { type: Number, default: 0 },
    lastApiCall: Date
  }
}, {
  timestamps: true
});

userSchema.pre('save', async function(next) {
  if (!this.isModified('password')) return next();
  
  try {
    const salt = await bcrypt.genSalt(12);
    this.password = await bcrypt.hash(this.password, salt);
    next();
  } catch (error) {
    next(error);
  }
});

userSchema.methods.comparePassword = async function(candidatePassword) {
  return bcrypt.compare(candidatePassword, this.password);
};

userSchema.methods.generateAuthToken = function() {
  return jwt.sign(
    { id: this._id, email: this.email, role: this.role },
    process.env.JWT_SECRET,
    { expiresIn: process.env.JWT_EXPIRES_IN }
  );
};

module.exports = mongoose.model('User', userSchema);

---

// models/Document.js - Document Model
const mongoose = require('mongoose');

const documentSchema = new mongoose.Schema({
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  filename: {
    type: String,
    required: true
  },
  originalName: {
    type: String,
    required: true
  },
  mimeType: String,
  size: Number,
  content: {
    type: String,
    required: true
  },
  chunks: [{
    index: Number,
    content: String,
    embedding: [Number]
  }],
  source: {
    type: String,
    enum: ['upload', 'jira', 'confluence', 'sharepoint'],
    default: 'upload'
  },
  metadata: {
    projectKey: String,
    spaceKey: String,
    driveId: String,
    extractedAt: Date
  },
  status: {
    type: String,
    enum: ['processing', 'completed', 'failed'],
    default: 'processing'
  }
}, {
  timestamps: true
});

documentSchema.index({ userId: 1, createdAt: -1 });
documentSchema.index({ source: 1, status: 1 });

module.exports = mongoose.model('Document', documentSchema);

---

// models/UserStory.js - User Story Model
const mongoose = require('mongoose');

const userStorySchema = new mongoose.Schema({
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  projectId: String,
  title: {
    type: String,
    required: true
  },
  description: {
    type: String,
    required: true
  },
  acceptanceCriteria: [String],
  priority: {
    type: String,
    enum: ['Critical', 'High', 'Medium', 'Low'],
    default: 'Medium'
  },
  storyPoints: {
    type: Number,
    min: 1,
    max: 21
  },
  epic: String,
  labels: [String],
  components: [String],
  status: {
    type: String,
    enum: ['generated', 'exported', 'archived'],
    default: 'generated'
  },
  jiraIssueKey: String,
  generationMetadata: {
    llmProvider: String,
    model: String,
    temperature: Number,
    tokensUsed: Number,
    processingTime: Number,
    sourceDocuments: [String]
  }
}, {
  timestamps: true
});

userStorySchema.index({ userId: 1, projectId: 1 });
userStorySchema.index({ status: 1, createdAt: -1 });

module.exports = mongoose.model('UserStory', userStorySchema);

---

// services/VectorService.js - Enhanced Vector Store with Weaviate
const weaviate = require('weaviate-ts-client');

class VectorService {
  constructor() {
    this.client = weaviate.client({
      scheme: 'http',
      host: process.env.WEAVIATE_HOST || 'localhost:8080',
    });
    this.className = 'Document';
  }

  async initialize() {
    try {
      // Create schema if it doesn't exist
      const schema = {
        class: this.className,
        description: 'Document chunks for RAG',
        properties: [
          {
            name: 'content',
            dataType: ['text'],
            description: 'The content of the document chunk'
          },
          {
            name: 'source',
            dataType: ['string'],
            description: 'Source of the document'
          },
          {
            name: 'title',
            dataType: ['string'],
            description: 'Document title'
          },
          {
            name: 'userId',
            dataType: ['string'],
            description: 'User ID who owns the document'
          },
          {
            name: 'chunkIndex',
            dataType: ['int'],
            description: 'Index of the chunk in the document'
          }
        ],
        vectorizer: 'text2vec-openai',
        moduleConfig: {
          'text2vec-openai': {
            model: 'ada',
            modelVersion: '002',
            type: 'text'
          }
        }
      };

      await this.client.schema.classCreator().withClass(schema).do();
    } catch (error) {
      if (!error.message.includes('already exists')) {
        console.error('Error initializing vector store:', error);
        throw error;
      }
    }
  }

  async addDocuments(documents, userId) {
    try {
      const batcher = this.client.batch.objectsBatcher();

      for (const doc of documents) {
        const chunks = this.chunkText(doc.content);
        
        for (let i = 0; i < chunks.length; i++) {
          batcher.withObject({
            class: this.className,
            properties: {
              content: chunks[i],
              source: doc.source,
              title: doc.title || doc.name,
              userId: userId.toString(),
              chunkIndex: i
            }
          });
        }
      }

      await batcher.do();
    } catch (error) {
      console.error('Error adding documents to vector store:', error);
      throw error;
    }
  }

  async search(query, userId, limit = 5) {
    try {
      const result = await this.client.graphql
        .get()
        .withClassName(this.className)
        .withFields('content source title chunkIndex _additional { certainty }')
        .withNearText({ concepts: [query] })
        .withWhere({
          path: ['userId'],
          operator: 'Equal',
          valueString: userId.toString()
        })
        .withLimit(limit)
        .do();

      return result.data.Get[this.className] || [];
    } catch (error) {
      console.error('Error searching vector store:', error);
      throw error;
    }
  }

  chunkText(text, maxChunkSize = 1000, overlap = 200) {
    const chunks = [];
    const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);
    
    let currentChunk = '';
    let currentSize = 0;
    
    for (const sentence of sentences) {
      const sentenceSize = sentence.length;
      
      if (currentSize + sentenceSize > maxChunkSize && currentChunk) {
        chunks.push(currentChunk.trim());
        
        // Create overlap
        const words = currentChunk.split(' ');
        const overlapWords = words.slice(-Math.floor(overlap / 10));
        currentChunk = overlapWords.join(' ') + ' ';
        currentSize = currentChunk.length;
      }
      
      currentChunk += sentence + '. ';
      currentSize += sentenceSize + 2;
    }
    
    if (currentChunk.trim()) {
      chunks.push(currentChunk.trim());
    }
    
    return chunks;
  }
}

module.exports = new VectorService();

---

// services/LLMService.js - Enhanced LLM Service
class LLMService {
  constructor() {
    this.clients = {};
    this.initializeClients();
  }

  initializeClients() {
    // OpenAI
    if (process.env.OPENAI_API_KEY) {
      this.clients.openai = new OpenAI({
        apiKey: process.env.OPENAI_API_KEY,
      });
    }

    // Azure OpenAI
    if (process.env.AZURE_OPENAI_ENDPOINT) {
      this.clients.azure = new OpenAI({
        apiKey: process.env.AZURE_OPENAI_API_KEY,
        baseURL: `${process.env.AZURE_OPENAI_ENDPOINT}/openai/deployments/${process.env.AZURE_OPENAI_DEPLOYMENT}`,
        defaultQuery: { 'api-version': process.env.AZURE_OPENAI_API_VERSION },
        defaultHeaders: {
          'api-key': process.env.AZURE_OPENAI_API_KEY,
        },
      });
    }

    // Anthropic Claude
    if (process.env.ANTHROPIC_API_KEY) {
      this.clients.claude = new Anthropic({
        apiKey: process.env.ANTHROPIC_API_KEY,
      });
    }
  }

  async generateUserStories(context, requirements, config, userId) {
    const client = this.clients[config.provider];
    if (!client) {
      throw new Error(`LLM provider ${config.provider} not configured`);
    }

    const startTime = Date.now();
    let tokensUsed = 0;

    try {
      const prompt = this.buildEnhancedPrompt(context, requirements);
      
      let response;
      
      if (config.provider === 'claude') {
        response = await client.messages.create({
          model: config.model,
          max_tokens: config.maxTokens || 4000,
          temperature: config.temperature,
          messages: [
            {
              role: 'user',
              content: prompt
            }
          ]
        });
        tokensUsed = response.usage.input_tokens + response.usage.output_tokens;
      } else {
        response = await client.chat.completions.create({
          model: config.model,
          messages: [
            {
              role: 'system',
              content: 'You are an expert product manager and business analyst. Generate well-structured user stories from requirements.'
            },
            {
              role: 'user',
              content: prompt
            }
          ],
          temperature: config.temperature,
          max_tokens: config.maxTokens || 4000
        });
        tokensUsed = response.usage.total_tokens;
      }

      const processingTime = Date.now() - startTime;
      
      // Update user usage statistics
      await this.updateUserUsage(userId, tokensUsed);

      const content = config.provider === 'claude' 
        ? response.content[0].text 
        : response.choices[0].message.content;

      const userStories = this.parseUserStories(content);
      
      // Add generation metadata
      userStories.forEach(story => {
        story.generationMetadata = {
          llmProvider: config.provider,
          model: config.model,
          temperature: config.temperature,
          tokensUsed,
          processingTime
        };
      });

      return userStories;
    } catch (error) {
      console.error('Error generating user stories:', error);
      throw error;
    }
  }

  buildEnhancedPrompt(context, requirements) {
    return `
You are an expert product manager with 15+ years of experience in agile development and user story creation.

CONTEXT FROM DOCUMENTS:
${context}

SPECIFIC REQUIREMENTS:
${requirements.join('\n')}

Generate comprehensive user stories following these guidelines:

1. USER STORY FORMAT:
   - Use "As a [persona], I want [functionality] so that [benefit]" format
   - Be specific about user personas (end user, admin, analyst, etc.)
   - Focus on user value and business outcomes

2. ACCEPTANCE CRITERIA:
   - 3-5 specific, testable criteria per story
   - Use "Given/When/Then" format where appropriate
   - Include edge cases and error handling

3. ESTIMATION & PRIORITIZATION:
   - Story points: 1, 2, 3, 5, 8, 13 (Fibonacci sequence)
   - Priority: Critical, High, Medium, Low
   - Consider complexity, risk, and business value

4. ADDITIONAL METADATA:
   - Group related stories under epics
   - Add relevant labels and components
   - Include dependencies where applicable

5. QUALITY STANDARDS:
   - Stories should be INVEST compliant (Independent, Negotiable, Valuable, Estimable, Small, Testable)
   - Avoid technical jargon in user-facing stories
   - Ensure stories are deliverable within a sprint

Return ONLY valid JSON in this exact format:
{
  "userStories": [
    {
      "id": "US-001",
      "title": "Concise story title",
      "description": "As a [persona], I want [functionality] so that [benefit]",
      "acceptanceCriteria": [
        "Given [context], when [action], then [outcome]",
        "The system should [requirement]",
        "Error handling: [scenario]"
      ],
      "priority": "High",
      "storyPoints": 5,
      "epic": "Epic name",
      "labels": ["frontend", "authentication"],
      "components": ["user-management", "api"],
      "dependencies": ["US-002"],
      "businessValue": "High|Medium|Low",
      "technicalRisk": "High|Medium|Low"
    }
  ]
}

Generate 8-12 comprehensive user stories that cover the main functional areas.`;
  }

  parseUserStories(content) {
    try {
      // Extract JSON from the response
      const jsonMatch = content.match(/\{[\s\S]*\}/);
      if (!jsonMatch) {
        throw new Error('No JSON found in response');
      }
      
      const parsed = JSON.parse(jsonMatch[0]);
      return parsed.userStories || [];
    } catch (error) {
      console.error('Error parsing user stories:', error);
      return this.fallbackParse(content);
    }
  }

  fallbackParse(content) {
    // Enhanced fallback parser
    const stories = [];
    const lines = content.split('\n').map(line => line.trim()).filter(line => line);
    
    let currentStory = null;
    let section = null;
    
    for (const line of lines) {
      if (line.match(/^(As a|As an)/i)) {
        if (currentStory) {
          stories.push(currentStory);
        }
        currentStory = {
          id: `US-${(stories.length + 1).toString().padStart(3, '0')}`,
          title: this.extractTitle(line),
          description: line,
          acceptanceCriteria: [],
          priority: 'Medium',
          storyPoints: 5,
          epic: '',
          labels: [],
          components: []
        };
        section = 'description';
      } else if (line.match(/acceptance criteria/i)) {
        section = 'criteria';
      } else if (line.match(/^(given|when|then|-|\*)/i) && section === 'criteria' && currentStory) {
        const criteria = line.replace(/^(-|\*)\s*/, '');
        currentStory.acceptanceCriteria.push(criteria);
      } else if (line.match(/priority:/i) && currentStory) {
        const priority = line.split(':')[1]?.trim();
        if (priority) currentStory.priority = priority;
      } else if (line.match(/story points?:/i) && currentStory) {
        const points = parseInt(line.split(':')[1]?.trim());
        if (points) currentStory.storyPoints = points;
      }
    }
    
    if (currentStory) {
      stories.push(currentStory);
    }
    
    return stories;
  }

  extractTitle(description) {
    // Extract a concise title from the user story description
    const match = description.match(/I want (.*?) so that/i);
    if (match) {
      return match[1].trim().replace(/^to /, '');
    }
    return description.substring(0, 50) + '...';
  }

  async updateUserUsage(userId, tokensUsed) {
    try {
      const User = require('../models/User');
      await User.findByIdAndUpdate(userId, {
        $inc: {
          'usage.apiCalls': 1,
          'usage.tokensUsed': tokensUsed,
          'usage.storiesGenerated': 1
        },
        $set: {
          'usage.lastApiCall': new Date()
        }
      });
    } catch (error) {
      console.error('Error updating user usage:', error);
    }
  }
}

module.exports = new LLMService();