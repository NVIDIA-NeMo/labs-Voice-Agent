<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Contributing To NeMo Voice Agent

## Development Environment

### Prerequisites

- Python 3.10 or higher (up to 3.13)
- [UV](https://github.com/astral-sh/uv) for fast Python package management
- Git for version control

### Setup

1. **Install UV**

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone the repository**

   ```bash
   git clone <repository-url>
   ```

3. **Set up development environment**

   ```bash
   uv sync --all-extras
   ```

4. **Install pre-commit hooks**

   ```bash
   uv run pre-commit install
   ```

### Development Tools

The project uses the following tools for development:

- **UV**: Fast Python package installer and resolver
- **Ruff**: Code formatting and linting
- **pytest**: Testing framework
- **pre-commit**: Git hooks for code quality

## Code Style and Quality

### Formatting

We use Ruff for consistent code formatting and linting:

```bash
# Format code manually
uv run ruff format .
uv run ruff check . --fix

# Or run pre-commit to format and check everything
uv run pre-commit run --all-files
```

### Code Quality Guidelines

1. **Follow Python conventions**: Use PEP 8 style guidelines (enforced by the linting tool above)
2. **Type annotations**: All public APIs must have complete type annotations
3. **Documentation**: Add docstrings for all public classes and methods
4. **Error handling**: Use explicit error handling with appropriate exception types
5. **Testing**: Write tests for new functionality and bug fixes
6. **Logging**: Use project logging facilities instead of print statements

## Testing

### Running Tests

```bash
# Run all tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=src --cov-report=term-missing

# Run specific test file
uv run pytest tests/test_specific_module.py
```

### Test Structure

- Unit tests are located in the `tests/` directory
- Test files should follow the naming pattern `test_*.py`
- Use descriptive test names that explain what is being tested
- Group related tests in classes when appropriate
- Mock external dependencies and network calls

### Writing Tests

1. **Test coverage**: Aim for high test coverage, especially for core functionality
2. **Test isolation**: Each test should be independent and not rely on others
3. **Clear assertions**: Use descriptive assertion messages
4. **Mock external dependencies**: Use `pytest` fixtures and mocking for external services

## Pull Request Guidelines

### Before Submitting

1. **Create an issue**: For significant changes, create an issue first to discuss the approach
2. **Branch naming**: Use descriptive branch names (e.g., `feature/add-new-exporter`, `fix/memory-leak`)
3. **Code quality**: Ensure all checks pass:

   ```bash
   uv run pre-commit run --all-files
   uv run pytest
   ```

### PR Requirements

1. **Clear description**: Explain what the PR does and why
2. **Tests**: Include tests for new functionality or bug fixes
3. **Documentation**: Update documentation if needed
4. **Backwards compatibility**: Avoid breaking changes unless necessary

### PR Process

1. **Fork the repository**: External contributors should fork the repository to their GitHub account
2. **Clone and branch**: Clone your fork and create a feature branch from `main`
3. **Make changes**: Implement your changes with tests
4. **Test thoroughly**: Run the full test suite
5. **Push to fork**: Push your branch to your forked repository
6. **Submit PR**: Create a pull request from your fork's branch to the main repository
7. **Address feedback**: Respond to review comments
8. **Squash commits**: Clean up commit history before merging

## Adding New Features

### Commit Message Format

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

**Types:**

- `feat`: New features
- `fix`: Bug fixes
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

### Getting Help

- **Documentation**: Check the README and inline documentation
- **Issues**: Search existing issues before creating new ones
- **Discussions**: Use GitHub discussions for questions
- **Code Review**: Ask for help in pull request comments

## Communication

- **Issues**: Use GitHub issues for bug reports and feature requests
- **Discussions**: Use GitHub discussions for questions and general discussion
- **Pull Requests**: Use PRs for code contributions with clear descriptions

## Signing Your Work

* We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

  * Any contribution which contains commits that are not Signed-Off will not be accepted.

* To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:
  ```bash
  $ git commit -s -m "Add cool feature."
  ```
  This will append the following to your commit message:
  ```
  Signed-off-by: Your Name <your@email.com>
  ```

* Full text of the DCO:

  ```
  Developer Certificate of Origin
  Version 1.1

  Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

  Everyone is permitted to copy and distribute verbatim copies of this
  license document, but changing it is not allowed.


  Developer's Certificate of Origin 1.1

  By making a contribution to this project, I certify that:

  (a) The contribution was created in whole or in part by me and I
      have the right to submit it under the open source license
      indicated in the file; or

  (b) The contribution is based upon previous work that, to the best
      of my knowledge, is covered under an appropriate open source
      license and I have the right under that license to submit that
      work with modifications, whether created in whole or in part
      by me, under the same open source license (unless I am
      permitted to submit under a different license), as indicated
      in the file; or

  (c) The contribution was provided directly to me by some other
      person who certified (a), (b) or (c) and I have not modified
      it.

  (d) I understand and agree that this project and the contribution
      are public and that a record of the contribution (including all
      personal information I submit with it, including my sign-off) is
      maintained indefinitely and may be redistributed consistent with
      this project or the open source license(s) involved.
  ```

## 🚀 Running GitHub CI

There are two ways to trigger CI tests on your pull request:

### Automatic CI Triggering

If your GitHub user is configured to use [signed commits](https://docs.github.com/en/authentication/managing-commit-signature-verification/about-commit-signature-verification), CI tests will run automatically when you push commits to your pull request.

> **Note**: Signed commits are different from signing-off on commits (which uses the `-s` flag mentioned in the [Signing Your Work](#signing-your-work) section).

### Manual CI Triggering

If you don't have signed commits set up, you can still trigger CI tests manually by commenting on your pull request:

```
/ok to test <commit-SHA>
```

For example:

```
/ok to test a1b2c3d4e5f6
```

**Important**: You'll need to add this comment for each new commit you push to ensure CI tests run on the latest changes.

#### Finding Your Commit SHA

You can find the commit SHA in several ways:

- View your pull request's commit history on GitHub
- Run `git log --oneline -1` in your local repository
- Check the commit details in your Git client
