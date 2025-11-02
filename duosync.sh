#!/bin/bash

# Script to synchronize the main branch to GitLab (origin) and the public-main branch to GitHub (github)

# Configuration
GITLAB_REMOTE="origin"
GITHUB_REMOTE="github"
MAIN_BRANCH="main"
PUBLIC_BRANCH="public-main"

# --- Helper Functions ---

# Function to check the last command's exit status
check_status() {
    if [ $? -ne 0 ]; then
        echo "ERROR: $1 failed. Aborting synchronization."
        exit 1
    fi
}

# --- Main Synchronization Logic ---

echo "Starting dual-remote synchronization..."

# 1. Ensure we are on the main branch
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$CURRENT_BRANCH" != "$MAIN_BRANCH" ]; then
    echo "Switching to $MAIN_BRANCH branch..."
    git checkout $MAIN_BRANCH
    check_status "git checkout $MAIN_BRANCH"
fi

# 2. Sync the main branch with GitLab (origin)
echo "1/3: Syncing $MAIN_BRANCH with $GITLAB_REMOTE (GitLab)..."
git pull --no-rebase $GITLAB_REMOTE $MAIN_BRANCH
check_status "git pull --no-rebase $GITLAB_REMOTE $MAIN_BRANCH"
git push $GITLAB_REMOTE $MAIN_BRANCH
check_status "git push $GITLAB_REMOTE $MAIN_BRANCH"
echo "GitLab sync complete."

# 3. Update and push the public-main branch to GitHub (github)
echo "2/3: Updating $PUBLIC_BRANCH with $MAIN_BRANCH changes..."
git checkout $PUBLIC_BRANCH
check_status "git checkout $PUBLIC_BRANCH"

# Rebase public-main onto main to get all new commits (excluding sensitive files due to git rm --cached)
git rebase $MAIN_BRANCH
check_status "git rebase $MAIN_BRANCH"

echo "3/3: Force-pushing $PUBLIC_BRANCH to $GITHUB_REMOTE (GitHub)..."
# Use --force to overwrite the remote branch history after rebase (necessary for clean exclusion)
git push --force $GITHUB_REMOTE $PUBLIC_BRANCH
check_status "git push --force $GITHUB_REMOTE $PUBLIC_BRANCH"
echo "GitHub sync complete."

# 4. Switch back to the main branch
echo "Switching back to $MAIN_BRANCH..."
git checkout $MAIN_BRANCH
check_status "git checkout $MAIN_BRANCH"

echo "Synchronization complete. Both remotes are up to date."