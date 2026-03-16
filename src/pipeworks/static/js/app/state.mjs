export const MAX_BATCH_SIZE = 1000;
export const COPY_FEEDBACK_MS = 1200;
export const SECTION_COLLAPSE_STORAGE_PREFIX = "pw-section-collapsed:";
export const DEFAULT_MODEL_ID = "flux-2-klein-4b";
export const PROMPT_SECTIONS = ["subject", "setting", "details", "lighting", "atmosphere"];
export const PROMPT_SECTION_LABELS = {
  subject: "subject",
  setting: "setting",
  details: "details",
  lighting: "lighting",
  atmosphere: "atmosphere",
};

export function emptyTokenCounts() {
  return {
    subject: 0,
    setting: 0,
    details: 0,
    lighting: 0,
    atmosphere: 0,
    total: 0,
    method: "heuristic",
  };
}

export const State = {
  config: null,
  runtimeMode: null,
  runtimeAuth: null,
  policyPromptOptions: [],
  policyPromptGroups: [],
  selectedModel: null,
  selectedGpuWorkerId: null,
  sectionModes: {
    subject: "manual",
    setting: "manual",
    details: "manual",
    lighting: "manual",
    atmosphere: "manual",
  },
  batchSize: 1,
  isGenerating: false,
  currentGenerationId: null,
  stopRequested: false,
  outputImages: [],
  galleryImages: [],
  favouriteImages: [],
  galleryPage: 1,
  galleryPerPage: 20,
  galleryPages: 1,
  favPage: 1,
  lightboxContext: null,
  theme: "dark",
  tokenCounts: emptyTokenCounts(),
  selectMode: false,
  selectedIds: new Set(),
  outputSelectMode: false,
  outputSelectedIds: new Set(),
};
