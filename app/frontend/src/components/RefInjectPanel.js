import React, { useState, useRef } from 'react';
import {
    Box,
    Typography,
    Button,
    Stack,
    TextField,
    ToggleButton,
    ToggleButtonGroup,
    Slider,
    Accordion,
    AccordionSummary,
    AccordionDetails,
    LinearProgress,
    Select,
    MenuItem,
} from '@mui/material';
import { Upload as UploadIcon, Square as StopIcon, ChevronDown as ExpandMoreIcon } from 'lucide-react';
import api from '../api';
import Tooltip from './Tooltip';
import { TIPS } from '../tooltips';

const SA3_LAYERS = 12;

export default function RefInjectPanel({ model_id, negativePrompt, loraStack, steps, cfgScale, samplerType = 'euler', distShift = 'none', onGenerated }) {
    const [prompt, setPrompt] = useState('');
    const [duration, setDuration] = useState(10);
    const [refAudioPath, setRefAudioPath] = useState('');
    const [refAudioName, setRefAudioName] = useState('');
    const [mode, setMode] = useState('inject');
    const [stepTaper, setStepTaper] = useState('cosine');
    const [timeTaper, setTimeTaper] = useState('none');
    const [layerStrengths, setLayerStrengths] = useState(() =>
        Array.from({ length: SA3_LAYERS }, () => 1.0)
    );
    const [generating, setGenerating] = useState(false);
    const [error, setError] = useState(null);
    const [progress, setProgress] = useState(0);
    const abortRef = useRef(null);
    const stopRef = useRef(false);

    const isDistilledBase = !!model_id && model_id.startsWith('sa3-') && !model_id.endsWith('-base');

    const handleLayerStrength = (idx) => (_, val) => {
        setLayerStrengths(prev => {
            const next = [...prev];
            next[idx] = val;
            return next;
        });
    };

    const handleUpload = async (e) => {
        const file = e.target.files?.[0];
        if (!file) return;
        try {
            const formData = new FormData();
            formData.append('file', file);
            const resp = await api.post('/api/audio/upload', formData);
            setRefAudioPath(resp.data.path);
            setRefAudioName(file.name);
        } catch (err) {
            setError('Upload failed: ' + (err.response?.data?.error?.message || err.message));
        }
    };

    const generate = async () => {
        if (!model_id) { setError('Pick a model in the Generation tab first.'); return; }
        if (!refAudioPath) { setError('Upload a reference audio clip first.'); return; }
        if (!prompt.trim()) { setError('Enter a prompt describing the target.'); return; }

        setGenerating(true);
        setError(null);
        setProgress(0);
        stopRef.current = false;
        const controller = new AbortController();
        abortRef.current = controller;

        let progressInterval;
        const startTicker = () => {
            progressInterval = setInterval(async () => {
                try {
                    const r = await api.get('/api/generation-progress');
                    const pct = Number(r.data?.progress) || 0;
                    setProgress(prev => Math.max(prev, Math.min(95, pct)));
                } catch {}
            }, 250);
        };
        const stopTicker = () => {
            if (progressInterval) { clearInterval(progressInterval); progressInterval = null; }
        };

        try {
            startTicker();

            const activeLoras = (loraStack || []).filter(s => s.path);
            const body = {
                model_id,
                prompt: prompt.trim(),
                duration,
                steps,
                ref_audio_path: refAudioPath,
                ref_inject_mode: mode,
                sampler_type: samplerType,
                dist_shift: distShift,
                ref_step_taper: stepTaper,
                ref_time_taper: timeTaper,
            };
            if (negativePrompt) body.negative_prompt = negativePrompt;
            if (!isDistilledBase) body.cfg_scale = cfgScale;
            if (activeLoras.length) {
                body.loras = activeLoras.map(s => ({
                    path: s.path,
                    strengths: s.bypassed
                        ? { sa: 0, ca: 0, mlp: 0 }
                        : (s.strengths || { sa: s.strength || 1.0, ca: s.strength || 1.0, mlp: s.strength || 1.0 }),
                }));
            }

            const strengthsMap = {};
            layerStrengths.forEach((v, i) => { if (v > 0) strengthsMap[i] = v; });
            if (Object.keys(strengthsMap).length) body.ref_layer_strengths = strengthsMap;

            const resp = await api.post('/api/generate', body, {
                responseType: 'blob',
                signal: controller.signal,
            });

            stopTicker();
            setProgress(100);

            const fname = resp.headers?.['x-fragment-filename'] || `ref_inject_${Date.now()}.wav`;
            const resolvedSeed = parseInt(resp.headers?.['x-fragment-seed'], 10);
            const params = Number.isFinite(resolvedSeed) ? { ...body, seed: resolvedSeed } : body;
            onGenerated?.(resp.data, fname, params);
        } catch (err) {
            if (err.name !== 'CanceledError') {
                setError(err.response?.data?.error?.message || err.message || 'Generation failed');
            }
        } finally {
            stopTicker();
            setGenerating(false);
        }
    };

    const stopGeneration = () => {
        stopRef.current = true;
        abortRef.current?.abort();
    };

    const gridItem = (label, tooltip, children) => (
        <Box sx={{ mb: 1.5 }}>
            <Tooltip title={tooltip}>
                <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 0.25, width: 'fit-content' }}>
                    {label}
                </Typography>
            </Tooltip>
            {children}
        </Box>
    );

    return (
        <Box sx={{ p: 2 }}>
            {error && (
                <Typography variant="caption" color="error" sx={{ display: 'block', mb: 1 }}>
                    {error}
                </Typography>
            )}

            {/* Prompt */}
            {gridItem('Generation Prompt', TIPS.generate.prompt,
                <TextField
                    fullWidth
                    multiline
                    minRows={1}
                    maxRows={3}
                    placeholder="Describe the target audio..."
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    size="small"
                />
            )}

            {/* Reference audio upload */}
            {gridItem('Reference Audio', 'Upload a reference clip whose self-attention patterns will guide generation (per-step KV injection).',
                refAudioPath ? (
                    <Stack direction="row" spacing={1} alignItems="center">
                        <Typography variant="body2" noWrap>{refAudioName}</Typography>
                        <Button size="small" variant="outlined" onClick={() => { setRefAudioPath(''); setRefAudioName(''); }}>
                            Clear
                        </Button>
                    </Stack>
                ) : (
                    <Button variant="outlined" size="small" component="label" startIcon={<UploadIcon size={16} />}>
                        Upload audio
                        <input type="file" hidden accept="audio/*" onChange={handleUpload} />
                    </Button>
                )
            )}

            {/* Duration */}
            {gridItem('Duration (seconds)', TIPS.generate.duration,
                <Stack direction="row" spacing={2} alignItems="center">
                    <Slider value={duration} onChange={(_, v) => setDuration(v)} min={1} max={120} step={1} valueLabelDisplay="auto" sx={{ flex: 1 }} />
                    <Typography variant="body2" color="text.secondary">{duration}s</Typography>
                </Stack>
            )}

            {/* Injection mode */}
            {gridItem('Injection Mode', 'How captured K/V from the reference are applied. Inject = blend with computed K/V. Replace = overwrite computed K/V. Threshold = only inject where attention scores exceed a threshold.',
                <ToggleButtonGroup value={mode} exclusive size="small" onChange={(_, v) => v && setMode(v)}>
                    <ToggleButton value="inject">Inject</ToggleButton>
                    <ToggleButton value="replace">Replace</ToggleButton>
                    <ToggleButton value="threshold">Threshold</ToggleButton>
                </ToggleButtonGroup>
            )}

            {/* Step taper */}
            {gridItem('Step Taper', 'Controls how injection strength scales across diffusion steps. Cosine = stronger early, fades out — recommended default.',
                <Select value={stepTaper} onChange={(e) => setStepTaper(e.target.value)} size="small" fullWidth>
                    <MenuItem value="none">None</MenuItem>
                    <MenuItem value="cosine">Cosine</MenuItem>
                    <MenuItem value="linear">Linear</MenuItem>
                </Select>
            )}

            {/* Time taper */}
            {gridItem('Time Taper', 'Controls how injection strength scales across the generated timeline (beginning vs end).',
                <Select value={timeTaper} onChange={(e) => setTimeTaper(e.target.value)} size="small" fullWidth>
                    <MenuItem value="none">None</MenuItem>
                    <MenuItem value="cosine">Cosine</MenuItem>
                    <MenuItem value="linear">Linear</MenuItem>
                </Select>
            )}

            {/* Per-layer strengths */}
            <Accordion defaultExpanded>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Typography variant="subtitle2">Per-Layer Strengths</Typography>
                </AccordionSummary>
                <AccordionDetails sx={{ maxHeight: 320, overflowY: 'auto', px: 1 }}>
                    {layerStrengths.map((val, i) => (
                        <Box key={i} sx={{ mb: 0.5 }}>
                            <Stack direction="row" spacing={1} alignItems="center">
                                <Typography variant="caption" sx={{ minWidth: 36 }}>L{i}</Typography>
                                <Slider
                                    value={val}
                                    onChange={handleLayerStrength(i)}
                                    min={0}
                                    max={3}
                                    step={0.05}
                                    size="small"
                                    sx={{ flex: 1 }}
                                />
                                <Typography variant="caption" sx={{ minWidth: 28, textAlign: 'right' }}>{val.toFixed(2)}</Typography>
                            </Stack>
                        </Box>
                    ))}
                </AccordionDetails>
            </Accordion>

            {/* Generate / Stop */}
            {generating ? (
                <Box sx={{ mt: 2 }}>
                    <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 0.5 }}>
                        <Typography variant="caption" color="text.secondary">Generating with reference injection... {Math.round(progress)}%</Typography>
                    </Stack>
                    <LinearProgress variant="determinate" value={Math.max(0, Math.min(100, progress))} sx={{ mb: 1 }} />
                    <Button variant="outlined" color="error" fullWidth startIcon={<StopIcon size={16} />} onClick={stopGeneration}>
                        Stop
                    </Button>
                </Box>
            ) : (
                <Button variant="contained" color="primary" fullWidth onClick={generate} sx={{ mt: 2 }}
                    disabled={!model_id || !prompt.trim() || !refAudioPath}>
                    Generate with Reference
                </Button>
            )}
        </Box>
    );
}
