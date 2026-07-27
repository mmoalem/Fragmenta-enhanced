import React, { useEffect, useState, useMemo } from 'react';
import {
    Box,
    Accordion,
    AccordionSummary,
    AccordionDetails,
    Button,
    Typography,
    Stack,
    MenuItem,
    Select,
    Slider,
    IconButton,
    Chip,
    Alert,
    Dialog,
    DialogTitle,
    DialogContent,
    Collapse,
    TextField,
    InputAdornment,
} from '@mui/material';
import { TIPS } from '../tooltips';
import Tooltip from './Tooltip';
import {
    Plus as AddIcon,
    Trash2 as RemoveIcon,
    GripVertical as DragIcon,
    Power as BypassIcon,
    ChevronDown as ChevronDownIcon,
    ChevronRight as ChevronRightIcon,
    Folder as FolderIcon,
    FolderOpen as FolderOpenIcon,
    ArrowUpDown as SortIcon,
    Search as SearchIcon,
} from 'lucide-react';
import api from '../api';
import { isLoraCompatible } from '../utils/loraMatch';

const MAX_SLOTS = 4;

/**
 * Multi-LoRA stack for the Generation panel.
 *
 * Props:
 *   selectedModel: the currently-selected base model id (e.g. "sa3-medium-base")
 *   value:         array of { path, strength, bypassed } slots
 *   onChange:      (newSlots) => void
 *
 * The picker filters available LoRAs by base-model compatibility (a `*-base`
 * LoRA also runs on its distilled sibling — see utils/loraMatch). Slot order
 * is the load order (slot 0 first); drag the handle to reorder. Bypass keeps
 * a slot in the stack but sends strength 0.
 */
export default function LoraStack({ selectedModel, value, onChange }) {
    const [available, setAvailable] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [dragIndex, setDragIndex] = useState(null);
    const [openFolder, setOpenFolder] = useState(null);
    const [pickerOpen, setPickerOpen] = useState(false);
    const [activeSlotIdx, setActiveSlotIdx] = useState(null);
    const [loraSort, setLoraSort] = useState('newest'); // 'newest' | 'alpha'
    const [loraSearch, setLoraSearch] = useState('');

    useEffect(() => {
        let cancelled = false;
        setLoading(true);
        api.get('/api/loras')
            .then(r => { if (!cancelled) setAvailable(r.data.loras || []); })
            .catch(e => { if (!cancelled) setError(e.response?.data?.error || e.message); })
            .finally(() => { if (!cancelled) setLoading(false); });
        return () => { cancelled = true; };
    }, []);

    // LoRAs compatible with the current generation model. A LoRA trained
    // against `*-base` is compatible with both that base and its distilled
    // sibling (same backbone, differ only in CFG state) — loraMatch strips
    // the trailing `-base` before comparing.
    const compatible = available.filter(l =>
        isLoraCompatible(l.base_model, selectedModel)
    );

    // Group compatible LoRAs by subfolder for the folder picker.
    // Root-level LoRAs (subfolder === '') go in a 'Root' group.
    const groupedLoRAs = useMemo(() => {
        // Apply search filter
        const query = loraSearch.trim().toLowerCase();
        const filtered = query
            ? compatible.filter(l => l.name.toLowerCase().includes(query))
            : compatible;

        // Apply sort
        const sorted = [...filtered].sort((a, b) => {
            if (loraSort === 'alpha') return a.name.localeCompare(b.name);
            return (b.mtime || 0) - (a.mtime || 0);
        });

        const groups = {};
        for (const lora of sorted) {
            const folder = lora.subfolder || 'Root';
            if (!groups[folder]) groups[folder] = [];
            groups[folder].push(lora);
        }
        // Sort folders: Root first, then alphabetical
        const result = Object.entries(groups).sort(([a], [b]) => {
            if (a === 'Root') return -1;
            if (b === 'Root') return 1;
            return a.localeCompare(b);
        });
        return result;
    }, [compatible, loraSort, loraSearch]);

    // The single-LoRA case stays one click: when no slots are populated AND
    // there's a compatible LoRA, surface one empty slot so the user sees a
    // "Pick a LoRA" dropdown immediately.
    const defaultSlot = { path: '', strengths: { sa: 1.0, ca: 1.0, mlp: 1.0 }, bypassed: false };
    const slots = (value && value.length > 0)
        ? value
        : (compatible.length ? [defaultSlot] : []);

    const addSlot = () => {
        if (slots.length >= MAX_SLOTS) return;
        onChange([...slots, { ...defaultSlot }]);
    };

    const removeSlot = (idx) => onChange(slots.filter((_, i) => i !== idx));

    const setSlot = (idx, patch) => {
        onChange(slots.map((s, i) => i === idx ? { ...s, ...patch } : s));
    };

    // --- drag-to-reorder (slot 0 is loaded first) ---------------------------
    const onDrop = (target) => {
        if (dragIndex === null || dragIndex === target) { setDragIndex(null); return; }
        const next = [...slots];
        const [moved] = next.splice(dragIndex, 1);
        next.splice(target, 0, moved);
        setDragIndex(null);
        onChange(next);
    };

    // --- folder picker handlers -------------------------------------------
    const openPicker = (idx) => {
        setPickerOpen(true);
        setActiveSlotIdx(idx);
        setOpenFolder(null);
        setLoraSearch('');
    };

    const closePicker = () => {
        setPickerOpen(false);
        setActiveSlotIdx(null);
        setOpenFolder(null);
    };

    const toggleFolder = (folder) => {
        setOpenFolder(prev => prev === folder ? null : folder);
    };

    const selectLora = (lora) => {
        if (activeSlotIdx !== null) {
            setSlot(activeSlotIdx, { path: lora.path });
        }
        closePicker();
    };

    const hint = (() => {
        if (!selectedModel) return 'Pick a model first.';
        if (loading) return 'Loading LoRAs…';
        if (!compatible.length) {
            return `No compatible LoRAs found for ${selectedModel}. Train one in the Training tab.`;
        }
        return null;
    })();

    return (
        <Accordion
            disableGutters
            defaultExpanded={Boolean(value && value.some((s) => s.path))}
        >
            <AccordionSummary expandIcon={<ChevronDownIcon size={18} />}>
                {/* Hover the title to surface the help in the Info View pill
                    (when it's on) — no inline "i", matching the rest of the app. */}
                <Tooltip title={TIPS.lora.stackInfo(MAX_SLOTS)}>
                    <Typography variant="subtitle1">LoRA Stack</Typography>
                </Tooltip>
            </AccordionSummary>
            <AccordionDetails
                onWheel={(e) => {
                    const el = e.currentTarget;
                    const atTop = el.scrollTop === 0;
                    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 2;
                    if ((e.deltaY < 0 && atTop) || (e.deltaY > 0 && atBottom)) return;
                    e.stopPropagation();
                }}
                sx={{ maxHeight: 400, overflow: 'auto' }}
            >
            {error && <Alert severity="error" sx={{ mb: 1 }}>{error}</Alert>}
            {hint && (
                <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
                    {hint}
                </Typography>
            )}

            {slots.length > 0 && (
                <Box sx={{ border: '1px solid', borderColor: 'divider', borderRadius: 1 }}>
                    {slots.map((slot, idx) => {
                        const choice = available.find(l => l.path === slot.path);
                        const bypassed = !!slot.bypassed;
                        return (
                            <Box
                                key={idx}
                                onDragOver={(e) => { if (dragIndex !== null) e.preventDefault(); }}
                                onDrop={() => onDrop(idx)}
                                sx={{
                                    p: 1.5,
                                    borderBottom: '1px solid',
                                    borderColor: 'divider',
                                    '&:last-child': { borderBottom: 'none' },
                                    bgcolor: dragIndex === idx ? 'action.hover' : 'transparent',
                                    opacity: bypassed ? 0.5 : 1,
                                }}
                            >
                                <Stack direction="row" alignItems="center" spacing={1}>
                                    <Tooltip title={TIPS.lora.dragReorder}>
                                        <Box
                                            draggable={slots.length > 1}
                                            onDragStart={() => setDragIndex(idx)}
                                            onDragEnd={() => setDragIndex(null)}
                                            sx={{
                                                display: 'flex',
                                                cursor: slots.length > 1 ? 'grab' : 'default',
                                                color: 'text.disabled',
                                            }}
                                        >
                                            <DragIcon size={16} />
                                        </Box>
                                    </Tooltip>
                                    <Typography variant="caption" color="text.disabled" sx={{ width: 14 }}>
                                        {idx}
                                    </Typography>
                                    <Tooltip title="Click to pick a LoRA">
                            <Button
                                size="small"
                                variant="outlined"
                                onClick={() => openPicker(idx)}
                                            sx={{
                                                flex: 1,
                                                minWidth: 0,
                                                justifyContent: 'flex-start',
                                                textTransform: 'none',
                                                textAlign: 'left',
                                            }}
                                        >
                                            {slot.path ? (
                                                <Typography variant="body2" noWrap sx={{ flex: 1 }}>
                                                    {choice ? `${choice.name} · ${choice.checkpoint}` : slot.path.split('/').pop()}
                                                </Typography>
                                            ) : (
                                                <Typography variant="body2" color="text.secondary">
                                                    Pick a LoRA
                                                </Typography>
                                            )}
                                        </Button>
                                    </Tooltip>
                                    <Tooltip title={TIPS.lora.bypass(bypassed)}>
                                        <IconButton
                                            size="small"
                                            color={bypassed ? 'default' : 'primary'}
                                            onClick={() => setSlot(idx, { bypassed: !bypassed })}
                                        >
                                            <BypassIcon size={14} />
                                        </IconButton>
                                    </Tooltip>
                                    <IconButton size="small" onClick={() => removeSlot(idx)} aria-label="Remove slot">
                                        <RemoveIcon size={14} />
                                    </IconButton>
                                </Stack>

                                <Stack direction="row" alignItems="center" spacing={1.5} sx={{ mt: 1, mb: 2 }}>
                                    <Tooltip title={TIPS.lora.sa}>
                                        <Typography variant="caption" color="text.secondary" sx={{ width: 60 }}>
                                            SA
                                        </Typography>
                                    </Tooltip>
                                    <Slider
                                        size="small"
                                        value={slot.strengths?.sa ?? 1.0}
                                        disabled={bypassed}
                                        onChange={(e, v) => setSlot(idx, { strengths: { ...slot.strengths, sa: v } })}
                                        min={-2}
                                        max={2}
                                        step={0.05}
                                        valueLabelDisplay="auto"
                                        marks={[
                                            { value: 0, label: '0' },
                                            { value: 1, label: '1' },
                                        ]}
                                        sx={{ flex: 1 }}
                                    />
                                    <Typography variant="body2" sx={{ width: 40, textAlign: 'right' }}>
                                        {bypassed ? '—' : (slot.strengths?.sa ?? 1.0).toFixed(2)}
                                    </Typography>
                                </Stack>

                                <Stack direction="row" alignItems="center" spacing={1.5} sx={{ mt: 1, mb: 2 }}>
                                    <Tooltip title={TIPS.lora.ca}>
                                        <Typography variant="caption" color="text.secondary" sx={{ width: 60 }}>
                                            CA
                                        </Typography>
                                    </Tooltip>
                                    <Slider
                                        size="small"
                                        value={slot.strengths?.ca ?? 1.0}
                                        disabled={bypassed}
                                        onChange={(e, v) => setSlot(idx, { strengths: { ...slot.strengths, ca: v } })}
                                        min={-2}
                                        max={2}
                                        step={0.05}
                                        valueLabelDisplay="auto"
                                        marks={[
                                            { value: 0, label: '0' },
                                            { value: 1, label: '1' },
                                        ]}
                                        sx={{ flex: 1 }}
                                    />
                                    <Typography variant="body2" sx={{ width: 40, textAlign: 'right' }}>
                                        {bypassed ? '—' : (slot.strengths?.ca ?? 1.0).toFixed(2)}
                                    </Typography>
                                </Stack>

                                <Stack direction="row" alignItems="center" spacing={1.5} sx={{ mt: 1, mb: 2 }}>
                                    <Tooltip title={TIPS.lora.mlp}>
                                        <Typography variant="caption" color="text.secondary" sx={{ width: 60 }}>
                                            MLP
                                        </Typography>
                                    </Tooltip>
                                    <Slider
                                        size="small"
                                        value={slot.strengths?.mlp ?? 1.0}
                                        disabled={bypassed}
                                        onChange={(e, v) => setSlot(idx, { strengths: { ...slot.strengths, mlp: v } })}
                                        min={-2}
                                        max={2}
                                        step={0.05}
                                        valueLabelDisplay="auto"
                                        marks={[
                                            { value: 0, label: '0' },
                                            { value: 1, label: '1' },
                                        ]}
                                        sx={{ flex: 1 }}
                                    />
                                    <Typography variant="body2" sx={{ width: 40, textAlign: 'right' }}>
                                        {bypassed ? '—' : (slot.strengths?.mlp ?? 1.0).toFixed(2)}
                                    </Typography>
                                </Stack>

                                {choice && choice.base_model && (
                                    <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.25 }}>
                                        Trained on {choice.base_model}
                                    </Typography>
                                )}
                            </Box>
                        );
                    })}
                </Box>
            )}

            <Stack direction="row" sx={{ mt: 1 }}>
                <Button
                    size="small"
                    variant="outlined"
                    startIcon={<AddIcon size={14} />}
                    disabled={slots.length >= MAX_SLOTS || !compatible.length}
                    onClick={addSlot}
                >
                    Add LoRA
                </Button>
            </Stack>

            {/* LoRA picker dialog — 80% of window, centered */}
            <Dialog
                open={pickerOpen}
                onClose={closePicker}
                maxWidth={false}
                PaperProps={{
                    sx: {
                        width: '80vw',
                        height: '80vh',
                        maxHeight: '80vh',
                        borderRadius: 2,
                    },
                }}
            >
                <DialogTitle sx={{ py: 1, px: 2, display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Typography variant="subtitle1" sx={{ flex: 1 }}>Pick a LoRA</Typography>
                </DialogTitle>
                <DialogContent sx={{ py: 0, px: 0, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
                    {/* Search bar */}
                    <Box sx={{ px: 2, pt: 1, pb: 1, display: 'flex', gap: 0.5, flexShrink: 0 }}>
                        <TextField
                            size="small"
                            placeholder="Search LoRAs…"
                            value={loraSearch}
                            onChange={(e) => setLoraSearch(e.target.value)}
                            autoFocus
                            InputProps={{
                                startAdornment: (
                                    <InputAdornment position="start">
                                        <SearchIcon size={14} />
                                    </InputAdornment>
                                ),
                                sx: { fontSize: 13, py: 0.25 },
                            }}
                            sx={{ flex: 1 }}
                        />
                        <Tooltip title={loraSort === 'newest' ? 'Newest first — click for alphabetical' : 'Alphabetical — click for newest first'}>
                            <IconButton
                                size="small"
                                onClick={() => setLoraSort(s => s === 'newest' ? 'alpha' : 'newest')}
                            >
                                <SortIcon size={14} />
                            </IconButton>
                        </Tooltip>
                    </Box>
                    {/* Scrollable list */}
                    <Box sx={{ flex: 1, overflow: 'auto', px: 1 }}>
                        {groupedLoRAs.length === 0 && (
                            <Typography variant="body2" color="text.secondary" sx={{ px: 2, py: 1 }}>
                                {loraSearch.trim() ? 'No matching LoRAs.' : 'No compatible LoRAs found.'}
                            </Typography>
                        )}
                        {groupedLoRAs.map(([folder, loras]) => (
                            <Box key={folder}>
                                {/* Folder header */}
                                <Button
                                    fullWidth
                                    size="small"
                                    onClick={() => toggleFolder(folder)}
                                    sx={{
                                        justifyContent: 'flex-start',
                                        textTransform: 'none',
                                        px: 1.5,
                                        py: 0.5,
                                        minHeight: 32,
                                        color: 'text.primary',
                                        '&:hover': { bgcolor: 'action.hover' },
                                    }}
                                >
                                    {openFolder === folder
                                        ? <FolderOpenIcon size={16} sx={{ mr: 1, color: 'text.secondary' }} />
                                        : <FolderIcon size={16} sx={{ mr: 1, color: 'text.secondary' }} />
                                    }
                                    <Typography variant="body2" sx={{ flex: 1, textAlign: 'left' }}>
                                        {folder === 'Root' ? 'Root (no folder)' : folder}
                                    </Typography>
                                    <Chip size="small" label={loras.length} sx={{ height: 18, fontSize: 10, mr: 0.5 }} />
                                    {openFolder === folder
                                        ? <ChevronDownIcon size={14} />
                                        : <ChevronRightIcon size={14} />
                                    }
                                </Button>

                                {/* Nested LoRA items */}
                                <Collapse in={openFolder === folder} timeout="auto">
                                    {loras.map((lora) => (
                                        <MenuItem
                                            key={lora.id}
                                            onClick={() => selectLora(lora)}
                                            sx={{ pl: 4, py: 0.5, minHeight: 36 }}
                                        >
                                            <Box sx={{ flex: 1, minWidth: 0 }}>
                                                <Typography variant="body2" noWrap>
                                                    {lora.name} · {lora.checkpoint}
                                                </Typography>
                                                <Stack direction="row" spacing={0.5} sx={{ mt: 0.25 }}>
                                                    <Chip size="small" label={lora.adapter_type || 'lora'} sx={{ height: 16, fontSize: 9 }} />
                                                    {lora.rank && <Chip size="small" label={`r=${lora.rank}`} sx={{ height: 16, fontSize: 9 }} />}
                                                </Stack>
                                            </Box>
                                        </MenuItem>
                                    ))}
                                </Collapse>
                            </Box>
                        ))}
                    </Box>
                </DialogContent>
            </Dialog>
            </AccordionDetails>
        </Accordion>
    );
}
