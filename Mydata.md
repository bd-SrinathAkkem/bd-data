Content A:

**1. Create Tag**
- Triggered if `$BD_VERSION` or `$CI_COMMIT_TAG` is set.
- Validates the version format and extracts components.
- Checks if the tag already exists:
  - If it exists, skips creation.
  - If not, creates an annotated tag and pushes it to the remote.

**2. Create Release**
- Triggered if `$BD_CREATE_RELEASE == "true"` and a valid version/tag is set.
- Ensures the tag exists.
- Checks if a release already exists for the tag:
  - If it exists, skips creation.
  - If not, creates a new release using the GitLab API.
- Handles prerelease and published release types.

**3. Sync Tags (latest, major)**
- Triggered if `$BD_SYNC_TAGS == "true"` and a valid version/tag is set.
- Skips if the tag is a prerelease or not in `vX.Y.Z` format.
- Fetches all tags and ensures the main tag exists.
- For each of `v$BD_MAJOR` and `latest`:
  - Deletes the tag locally and remotely if it exists.
  - Recreates the tag pointing to the current release commit.
  - Pushes the tag to the remote.

**General Flow**
- Each stage depends on the previous one via `needs` and artifact passing.
- Variables are exported and reused between jobs using dotenv artifacts.

**Pipeline Stages (in order):**
1. `bd_validate_version`
2. `bd_create_tag`
3. `bd_create_release`
4. `bd_sync_tags`

**How to trigger:**
- Set the appropriate variables (`BD_VERSION`, `BD_CREATE_RELEASE`, `BD_SYNC_TAGS`) in the pipeline or CI/CD variables.
- Push a commit or tag, or run the pipeline manually with the desired variables.

**Note:**  
All jobs use a retry mechanism and require a valid `CI_JOB_TOKEN` for Git and API operations.


Content B:

Code Freeze and Release

Create Tag

Using the semver_upgrade.yml Workflow:

Navigate to the CI/CD > Pipelines tab within your GitLab project.
Click on Run Pipeline.
Enter the following input:
VERSION=Enter the new version number (e.g., vX.Y.Z).
Click Run Pipeline.
This workflow will create a new tag and push it to the remote repository.
QA Approval

Submit the tagged release for QA approval via the SIGINT ticket.
Create a SIGINT Jira ticket assigned to the Security Champion (@Suhas LR).
Include links to scan results and SBOM from pipelines.
Publish to Marketplace

GitLab CI/CD Publishing

Begin by manually drafting a release for the marketplace and marking it as the latest version.
The sync-tags action workflow should automatically trigger.
If the workflow does not trigger, you can manually initiate it by following these steps:
Navigate to the CI/CD > Pipelines tab within your GitLab project.
Click on Run Pipeline.
Complete the following inputs:
SYNC_TAGS=true
Finally, click Run Pipeline.
This workflow will ensure that all tags are synchronized with the remote repository.

