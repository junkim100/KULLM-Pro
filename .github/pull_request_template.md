# Pull Request

## 📋 Description

**What does this PR do?**
A clear and concise description of what this pull request accomplishes.

**Related Issue(s):**
- Fixes #(issue number)
- Closes #(issue number)
- Related to #(issue number)

## 🔄 Type of Change

Please mark the relevant option(s):

- [ ] 🐛 Bug fix (non-breaking change which fixes an issue)
- [ ] ✨ New feature (non-breaking change which adds functionality)
- [ ] 💥 Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] 📚 Documentation update
- [ ] 🧹 Code refactoring (no functional changes)
- [ ] ⚡ Performance improvement
- [ ] 🧪 Test coverage improvement
- [ ] 🔧 Build/CI improvement

## 🧪 Testing

**How has this been tested?**
Please describe the tests that you ran to verify your changes:

- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] Code switching functionality tested
- [ ] Fine-tuning functionality tested

**Test Configuration:**
- Python version: [e.g. 3.9]
- OS: [e.g. Ubuntu 20.04]
- GPU: [e.g. NVIDIA RTX 4090, None]

**Test commands run:**
```bash
# List the commands you used to test
pytest tests/
python code_switch.py run "GAIR/LIMO" --split="train" --n=2
python fine_tune.py train --train_file="test.jsonl" --output_dir="test_output"
```

## 📝 Changes Made

**Detailed list of changes:**
- Added/Modified: [specific files and functions]
- Removed: [what was removed and why]
- Fixed: [what bugs were fixed]
- Improved: [what was optimized or enhanced]

**Code changes:**
- Files modified: [list key files]
- New files added: [list new files]
- Dependencies added/removed: [list any dependency changes]

## 📸 Screenshots (if applicable)

Add screenshots to help explain your changes, especially for UI/CLI changes.

## ✅ Checklist

**Before submitting this PR, please make sure:**

### Code Quality
- [ ] My code follows the project's style guidelines
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings

### Testing
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] I have tested the code switching functionality (if applicable)
- [ ] I have tested the fine-tuning functionality (if applicable)

### Documentation
- [ ] I have updated the README.md (if needed)
- [ ] I have updated the CHANGELOG.md
- [ ] I have added/updated docstrings for new functions
- [ ] I have updated configuration examples (if needed)

### Dependencies
- [ ] I have updated requirements.txt (if new dependencies were added)
- [ ] All new dependencies are justified and documented
- [ ] I have tested with the minimum required versions

## 🔍 Review Focus Areas

**Please pay special attention to:**
- [ ] Performance implications
- [ ] Security considerations
- [ ] API compatibility
- [ ] Error handling
- [ ] Memory usage
- [ ] Configuration changes

## 📚 Additional Notes

**Any additional information for reviewers:**
- Special considerations
- Known limitations
- Future improvements planned
- Migration notes (for breaking changes)

## 🤝 Reviewer Assignment

**Suggested reviewers:**
@username1 @username2

**Why these reviewers:**
- Expert in [specific area]
- Familiar with [relevant component]
- Requested review
