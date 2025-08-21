# Makefile - Easy Commands
.PHONY: help build up down logs clean test

help: ## Show this help
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

build: ## Build all services
	docker-compose build

up: ## Start all services in development mode
	docker-compose up -d

down: ## Stop all services
	docker-compose down

logs: ## View logs from all services
	docker-compose logs -f

clean: ## Clean up containers, networks, images, and volumes
	docker-compose down -v
	docker system prune -f

test: ## Run tests
	docker-compose exec backend npm test

prod-up: ## Start production environment
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

prod-down: ## Stop production environment
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml down

backup-db: ## Backup MongoDB
	docker-compose exec mongodb mongodump --out /backup/$(shell date +%Y%m%d_%H%M%S)

restore-db: ## Restore MongoDB (specify BACKUP_DIR)
	docker-compose exec mongodb mongorestore /backup/$(BACKUP_DIR)

shell-backend: ## Open shell in backend container
	docker-compose exec backend sh

shell-mongo: ## Open MongoDB shell
	docker-compose exec mongodb mongosh

shell-redis: ## Open Redis shell
	docker-compose exec redis redis-cli

---
