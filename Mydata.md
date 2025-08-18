Got it ✅ — instead of describing *what happens internally*, you want Content B rewritten into a **step-by-step “How To” guide** for running the release process (with the details from Content A baked in).

Here’s the restructured version:

---

# Code Freeze and Release Guide

This guide explains how to run the release process using GitLab CI/CD.

---

## **1. Create Tag**

1. Go to **CI/CD > Pipelines** in your GitLab project.
2. Click **Run Pipeline**.
3. In the variable input section, enter:

   * `BD_VERSION=vX.Y.Z` (for example, `v1.2.3`).
4. Run the pipeline.
5. The pipeline will:

   * Validate the version format.
   * Create an annotated tag if it does not already exist.
   * Push the tag to the remote repository.

---

## **2. Create Release**

1. Ensure you already created a tag (see step 1).
2. Go to **CI/CD > Pipelines**.
3. Click **Run Pipeline**.
4. Enter the variables:

   * `BD_VERSION=vX.Y.Z`
   * `BD_CREATE_RELEASE=true`
5. Run the pipeline.
6. The pipeline will:

   * Verify the tag exists.
   * Create a new release in GitLab if one does not exist already.
   * Handle prerelease or published release types automatically.

**QA Approval:**

* After the release is created, submit a **SIGINT Jira ticket**.
* Assign it to **@Suhas LR (Security Champion)**.
* Attach links to:

  * Pipeline scan results
  * SBOM (Software Bill of Materials)

---

## **3. Sync Tags (latest, major)**

1. Make sure the release tag is in `vX.Y.Z` format (not a prerelease).
2. Go to **CI/CD > Pipelines**.
3. Click **Run Pipeline**.
4. Enter the variables:

   * `BD_VERSION=vX.Y.Z`
   * `BD_SYNC_TAGS=true`
5. Run the pipeline.
6. The pipeline will:

   * Delete any existing `latest` and `vX` tags locally and remotely.
   * Recreate them pointing to the current release commit.
   * Push the tags to the remote repository.

---

## **Pipeline Stages (in order)**

1. `bd_validate_version`
2. `bd_create_tag`
3. `bd_create_release`
4. `bd_sync_tags`

Each stage depends on the previous one and passes variables using artifacts.

---

## **Notes**

* Always run the **Create Tag** step before **Create Release** or **Sync Tags**.
* Use the same version (`BD_VERSION`) consistently across steps.
* A valid `CI_JOB_TOKEN` is required for all Git and API operations.
* All jobs retry automatically if something transient fails.

---

👉 This version is focused on **How to run each step**.

Do you want me to make it even more **checklist-style** (like bullet points you can copy directly into a release runbook), or keep it as structured sections?
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

