# Use Node.js base image
FROM node:22-alpine

WORKDIR /app

# Copy package.json and install dependencies with legacy-peer-deps
COPY package*.json ./
RUN npm install --legacy-peer-deps

# Copy the rest of the frontend code
COPY . .

# Expose port
EXPOSE 3000

# Start React dev server
CMD ["npm", "start"]
