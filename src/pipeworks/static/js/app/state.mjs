export const MAX_BATCH_SIZE = 1000;
export const COPY_FEEDBACK_MS = 1200;
export const SECTION_COLLAPSE_STORAGE_PREFIX = "pw-section-collapsed:";
export const DEFAULT_MODEL_ID = "flux-2-klein-4b";
export const DEFAULT_SLOT_LABEL = "Policy";
export const PROMPT_SCHEMA_VERSION = 3;

let _slotIdCounter = 0;

export function nextSlotId() {
  _slotIdCounter += 1;
  return `slot-${_slotIdCounter}`;
}

export function defaultSlot() {
  return {
    id: nextSlotId(),
    label: DEFAULT_SLOT_LABEL,
    mode: "manual",
    manualText: "",
    selectedPolicyId: null,
    tokens: 0,
  };
}

export function emptyTokenCounts() {
  return {
    sections: [],
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
  sections: [defaultSlot()],
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
