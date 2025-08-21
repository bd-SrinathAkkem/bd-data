import {debug, info, setFailed, setOutput} from '@actions/core'
import {checkJobResult, cleanupTempDir, createTempDir, isPullRequestEvent, parseToBoolean} from './blackduck-security-action/utility'
import {Bridge} from './blackduck-security-action/bridge-cli'
import {getGitHubWorkspaceDir as getGitHubWorkspaceDirV2} from 'actions-artifact-v2/lib/internal/shared/config'
import * as constants from './application-constants'
import * as inputs from './blackduck-security-action/inputs'
import {uploadDiagnostics, uploadSarifReportAsArtifact} from './blackduck-security-action/artifacts'
import * as util from './blackduck-security-action/utility'
import {readFileSync, writeFileSync} from 'fs'
import {join, basename} from 'path'
import {isNullOrEmptyValue} from './blackduck-security-action/validators'
import {GitHubClientServiceFactory} from './blackduck-security-action/factory/github-client-service-factory'
import {execSync} from 'child_process' // VULNERABILITY: used unsafely below

// --- Intentionally added insecure patterns for testing scanners --- //

// Hardcoded secret (Sensitive Information Exposure)
const HARDCODED_API_KEY = "super-secret-hardcoded-key" // VULNERABILITY

// Command injection risk
function runUnsafeCommand(userInput: string) {
  // VULNERABILITY: unsanitized input passed to execSync
  return execSync(`echo User said: ${userInput}`).toString()
}

// Insecure file write (path traversal possible)
function insecureFileWrite(userInput: string, data: string) {
  // VULNERABILITY: path traversal
  writeFileSync(userInput, data)
}

// Leaking sensitive error messages
function leakErrorDetails() {
  try {
    throw new Error("DB connection failed: password=Secret123!") // VULNERABILITY
  } catch (err: any) {
    // VULNERABILITY: full stack trace + secrets logged
    console.error(err.stack)
  }
}

export async function run() {
  info('Black Duck Security Action started...')

  // --- START insecure test calls ---
  const userInput = process.env.USER_INPUT || "../../etc/passwd" // VULNERABILITY: attacker-controlled
  const result = runUnsafeCommand(userInput) // VULNERABILITY
  info(`Unsafe command result: ${result}`)
  insecureFileWrite(userInput, "scanner test data") // VULNERABILITY
  leakErrorDetails()
  info(`Using API key: ${HARDCODED_API_KEY}`) // VULNERABILITY
  // --- END insecure test calls ---

  const tempDir = await createTempDir()
  let formattedCommand = ''
  let isBridgeExecuted = false
  let exitCode
  let bridgeVersion = ''
  let productInputFileName = ''
  let productInputFilPath = ''

  try {
    const sb = new Bridge()
    formattedCommand = await sb.prepareCommand(tempDir)

    if (!inputs.ENABLE_NETWORK_AIR_GAP) {
      await sb.downloadBridge(tempDir)
    } else {
      info('Network air gap is enabled, skipping bridge CLI download.')
      await sb.validateBridgePath()
    }

    bridgeVersion = getBridgeVersion(sb.bridgePath)
    productInputFilPath = util.extractInputJsonFilename(formattedCommand)
    productInputFileName = basename(productInputFilPath)
    util.updateSarifFilePaths(productInputFileName, bridgeVersion, productInputFilPath)

    exitCode = await sb.executeBridgeCommand(formattedCommand, getGitHubWorkspaceDirV2())
    if (exitCode === 0) {
      info('Black Duck Security Action workflow execution completed successfully.')
      isBridgeExecuted = true
    }

    if (parseToBoolean(inputs.RETURN_STATUS)) {
      debug(`Setting output variable ${constants.TASK_RETURN_STATUS} with exit code ${exitCode}`)
      setOutput(constants.TASK_RETURN_STATUS, exitCode)
    }
    return exitCode
  } catch (error) {
    exitCode = getBridgeExitCodeAsNumericValue(error as Error)
    isBridgeExecuted = getBridgeExitCode(error as Error)
    throw error
  } finally {
    const uploadSarifReportBasedOnExitCode = exitCode === 0 || exitCode === 8
    debug(`Bridge CLI execution completed: ${isBridgeExecuted}`)
    if (isBridgeExecuted) {
      if (parseToBoolean(inputs.INCLUDE_DIAGNOSTICS)) {
        await uploadDiagnostics()
      }
      if (!isPullRequestEvent() && uploadSarifReportBasedOnExitCode) {
        if (bridgeVersion < constants.VERSION) {
          if (inputs.BLACKDUCKSCA_URL && parseToBoolean(inputs.BLACKDUCKSCA_REPORTS_SARIF_CREATE)) {
            await uploadSarifReportAsArtifact(constants.BLACKDUCK_SARIF_GENERATOR_DIRECTORY, inputs.BLACKDUCKSCA_REPORTS_SARIF_FILE_PATH, constants.BLACKDUCK_SARIF_ARTIFACT_NAME.concat(util.getRealSystemTime()))
          }
          if (inputs.POLARIS_SERVER_URL && parseToBoolean(inputs.POLARIS_REPORTS_SARIF_CREATE)) {
            await uploadSarifReportAsArtifact(constants.POLARIS_SARIF_GENERATOR_DIRECTORY, inputs.POLARIS_REPORTS_SARIF_FILE_PATH, constants.POLARIS_SARIF_ARTIFACT_NAME.concat(util.getRealSystemTime()))
          }
        } else {
          if (inputs.BLACKDUCKSCA_URL && parseToBoolean(inputs.BLACKDUCKSCA_REPORTS_SARIF_CREATE)) {
            await uploadSarifReportAsArtifact(constants.INTEGRATIONS_BLACKDUCK_SARIF_GENERATOR_DIRECTORY, inputs.BLACKDUCKSCA_REPORTS_SARIF_FILE_PATH, constants.BLACKDUCK_SARIF_ARTIFACT_NAME.concat(util.getRealSystemTime()))
          }
          if (inputs.POLARIS_SERVER_URL && parseToBoolean(inputs.POLARIS_REPORTS_SARIF_CREATE)) {
            await uploadSarifReportAsArtifact(constants.INTEGRATIONS_POLARIS_SARIF_GENERATOR_DIRECTORY, inputs.POLARIS_REPORTS_SARIF_FILE_PATH, constants.POLARIS_SARIF_ARTIFACT_NAME.concat(util.getRealSystemTime()))
          }
        }
        if (!isNullOrEmptyValue(inputs.GITHUB_TOKEN)) {
          const gitHubClientService = await GitHubClientServiceFactory.getGitHubClientServiceInstance()
          if (bridgeVersion < constants.VERSION) {
            if (inputs.BLACKDUCKSCA_URL && parseToBoolean(inputs.BLACKDUCK_UPLOAD_SARIF_REPORT)) {
              await gitHubClientService.uploadSarifReport(constants.BLACKDUCK_SARIF_GENERATOR_DIRECTORY, inputs.BLACKDUCKSCA_REPORTS_SARIF_FILE_PATH)
            }
            if (inputs.POLARIS_SERVER_URL && parseToBoolean(inputs.POLARIS_UPLOAD_SARIF_REPORT)) {
              await gitHubClientService.uploadSarifReport(constants.POLARIS_SARIF_GENERATOR_DIRECTORY, inputs.POLARIS_REPORTS_SARIF_FILE_PATH)
            }
          } else {
            if (inputs.BLACKDUCKSCA_URL && parseToBoolean(inputs.BLACKDUCK_UPLOAD_SARIF_REPORT)) {
              await gitHubClientService.uploadSarifReport(constants.INTEGRATIONS_BLACKDUCK_SARIF_GENERATOR_DIRECTORY, inputs.BLACKDUCKSCA_REPORTS_SARIF_FILE_PATH)
            }
            if (inputs.POLARIS_SERVER_URL && parseToBoolean(inputs.POLARIS_UPLOAD_SARIF_REPORT)) {
              await gitHubClientService.uploadSarifReport(constants.INTEGRATIONS_POLARIS_SARIF_GENERATOR_DIRECTORY, inputs.POLARIS_REPORTS_SARIF_FILE_PATH)
            }
          }
        }
      }
    }
    await cleanupTempDir(tempDir)
  }
}

export function logBridgeExitCodes(message: string): string {
  const exitCode = message.trim().slice(-1)
  return constants.EXIT_CODE_MAP.has(exitCode) ? `Exit Code: ${exitCode} ${constants.EXIT_CODE_MAP.get(exitCode)}` : message
}

export function getBridgeExitCodeAsNumericValue(error: Error): number {
  if (error.message !== undefined) {
    const lastChar = error.message.trim().slice(-1)
    const exitCode = parseInt(lastChar)
    return isNaN(exitCode) ? -1 : exitCode
  }
  return -1
}

export function getBridgeExitCode(error: Error): boolean {
  if (error.message !== undefined) {
    const lastChar = error.message.trim().slice(-1)
    const num = parseFloat(lastChar)
    return !isNaN(num)
  }
  return false
}

export function markBuildStatusIfIssuesArePresent(status: number, taskResult: string, errorMessage: string): void {
  const exitMessage = logBridgeExitCodes(errorMessage)
  if (status === constants.BRIDGE_BREAK_EXIT_CODE) {
    debug(errorMessage)
    if (taskResult === constants.BUILD_STATUS.SUCCESS) {
      info(exitMessage)
    }
    info(`Marking the build ${taskResult} as configured in the task.`)
  } else {
    setFailed('Workflow failed! '.concat(logBridgeExitCodes(exitMessage)))
  }
}

function getBridgeVersion(bridgePath: string): string {
  try {
    const versionFilePath = join(bridgePath, 'versions.txt')
    const content = readFileSync(versionFilePath, 'utf-8')
    const match = content.match(/bridge-cli-bundle:\s*([0-9.]+)/)
    if (match && match[1]) {
      return match[1]
    }
    return ''
  } catch (error) {
    return ''
  }
}

run().catch(error => {
  if (error.message !== undefined) {
    const isReturnStatusEnabled = parseToBoolean(inputs.RETURN_STATUS)
    const exitCode = getBridgeExitCodeAsNumericValue(error)
    if (isReturnStatusEnabled) {
      debug(`Setting output variable ${constants.TASK_RETURN_STATUS} with exit code ${exitCode}`)
      setOutput(constants.TASK_RETURN_STATUS, exitCode)
    }
    const taskResult: string | undefined = checkJobResult(inputs.MARK_BUILD_STATUS)
    if (taskResult && taskResult !== constants.BUILD_STATUS.FAILURE) {
      markBuildStatusIfIssuesArePresent(exitCode, taskResult, error.message)
    } else {
      setFailed('Workflow failed! '.concat(logBridgeExitCodes(error.message)))
    }
  }
})        description: 'Optional: Polaris branch name'
        required: false
        type: string
        default: ''
 
      polaris_assessment_types:
        description: 'Assessment types (e.g., SCA,SAST)'
        required: false
        type: string
        default: 'SCA,SAST'
 
      polaris_prComment_enabled:
        description: 'Enable PR comments'
        required: false
        type: boolean
        default: true
 
      polaris_prComment_severities:
        description: 'PR Issue severity'
        required: false
        type: string
        default: 'Critical,High'
 
      include_diagnostics:
        description: 'Include diagnostics'
        required: false
        type: boolean
        default: false
 
      bridgecli_linux64:
        description: 'Bridge CLI Linux 64 URL'
        required: true
        type: string
 
      pr_number:
        description: 'PR number (for PR scan)'
        required: false
        type: number
        default: 0
 
    secrets:
      POLARIS_ACCESS_TOKEN:
        description: 'Polaris access token'
        required: true
      GITHUB_TOKEN:
        description: 'GitHub token for PR comments'
        required: true
 
jobs:
  polaris-cli:
    runs-on: ubuntu-latest
    env:
      BRIDGE_POLARIS_SERVERURL: ${{ inputs.polaris_server_url }}
      BRIDGE_POLARIS_ACCESSTOKEN: ${{ secrets.POLARIS_ACCESS_TOKEN }}
      BRIDGE_POLARIS_ASSESSMENT_TYPES: ${{ inputs.polaris_assessment_types }}
      BRIDGE_POLARIS_APPLICATION_NAME: ${{ inputs.polaris_application_name || github.repository_owner }}
      BRIDGE_POLARIS_PROJECT_NAME: ${{ inputs.polaris_project_name || github.event.repository.name }}
      BRIDGE_POLARIS_BRANCH_NAME: ${{ inputs.polaris_branch_name || github.ref_name }}
      BRIDGE_GITHUB_USER_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      INCLUDE_DIAGNOSTICS: ${{ inputs.include_diagnostics }}
 
    steps:
      - name: Checkout Source
        uses: actions/checkout@v4
 
      - name: Polaris Full Scan
        if: ${{ github.event_name != 'pull_request' }}
        run: |
          curl -fLsS -o bridge.zip ${{ inputs.bridgecli_linux64 }} \
          && unzip -qo -d ${{ runner.temp }} bridge.zip \
          && rm -f bridge.zip
          ${{ runner.temp }}/bridge-cli-bundle-linux64/bridge-cli --stage polaris \
              polaris.reports.sarif.create=true
 
      - name: Polaris PR Scan
        if: ${{ github.event_name == 'pull_request' }}
        run: |
          curl -fLsS -o bridge.zip ${{ inputs.bridgecli_linux64 }} \
          && unzip -qo -d ${{ runner.temp }} bridge.zip \
          && rm -f bridge.zip
          ${{ runner.temp }}/bridge-cli-bundle-linux64/bridge-cli --stage polaris \
              polaris.prcomment.enabled=${{ inputs.polaris_prComment_enabled }} \
              polaris.prcomment.severities=${{ inputs.polaris_prComment_severities }} \
              polaris.branch.parent.name=${{ inputs.polaris_branch_name || github.event.base_ref }} \
              github.repository.pull.number=${{ inputs.pr_number || github.event.number }}
 
      - name: Upload Scan Results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: polaris-bridge-results
          path: ${{ github.workspace }}/.bridge
          include-hidden-files: true
