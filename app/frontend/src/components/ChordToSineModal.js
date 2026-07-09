import React, { useState, useRef, useEffect } from 'react';
import {
    Box, Button, Dialog, DialogActions, DialogContent, DialogTitle,
    Typography, Stack, Tabs, Tab, Slider, TextField, LinearProgress,
    Alert, IconButton,
} from '@mui/material';
import { Upload as UploadIcon, Music as MusicIcon, X as ClearIcon } from 'lucide-react';
import api from '../api';

/** Decode a Blob to an AudioBuffer via AudioContext. */
async function blobToAudioBuffer(ctx, blob) {
    const ab = await blob.arrayBuffer();
    return ctx.decodeAudioData(ab);
}

/** Encode a mono AudioBuffer to a 16-bit PCM WAV Blob. */
function audioBufferToWav(buf) {
    const sr = buf.sampleRate;
    const ch = buf.numberOfChannels;
    const len = buf.length;
    const ns = len * ch;
    const data = new ArrayBuffer(44 + ns * 2);
    const v = new DataView(data);
    const w = (off, str) => { for (let i = 0; i < str.length; i++) v.setUint8(off + i, str.charCodeAt(i)); };
    w(0, 'RIFF'); v.setUint32(4, 36 + ns * 2, true);
    w(8, 'WAVE'); w(12, 'fmt ');
    v.setUint32(16, 16, true); v.setUint16(20, 1, true);
    v.setUint16(22, ch, true); v.setUint32(24, sr, true);
    v.setUint32(28, sr * ch * 2, true);
    v.setUint16(32, ch * 2, true); v.setUint16(34, 16, true);
    w(36, 'data'); v.setUint32(40, ns * 2, true);
    for (let c = 0; c < ch; c++) {
        const chan = buf.getChannelData(c);
        for (let i = 0; i < len; i++) {
            const s = Math.max(-1, Math.min(1, chan[i]));
            v.setInt16(44 + (i * ch + c) * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
        }
    }
    return new Blob([data], { type: 'audio/wav' });
}

export default function ChordToSineModal({ open, onClose, onApplySource }) {
    const [tab, setTab] = useState(0);

    // ---- MIDI tab state ----
    const [midiFile, setMidiFile] = useState(null);
    const [midiRendering, setMidiRendering] = useState(false);
    const [midiAudioUrl, setMidiAudioUrl] = useState(null);
    const [midiDuration, setMidiDuration] = useState(null);
    const [midiPath, setMidiPath] = useState(null);
    const [midiWaveform, setMidiWaveform] = useState('sine');
    const [midiTranspose, setMidiTranspose] = useState(0);
    const [midiName, setMidiName] = useState('');
    const midiFileRef = useRef(null);

    // ---- Chord tab state ----
    const [chordAudioFile, setChordAudioFile] = useState(null);
    const [chordAudioUrl, setChordAudioUrl] = useState(null);
    const [chordText, setChordText] = useState('');
    const [chordExtracting, setChordExtracting] = useState(false);
    const [chordRendering, setChordRendering] = useState(false);
    const [chordResultUrl, setChordResultUrl] = useState(null);
    const [chordResultDuration, setChordResultDuration] = useState(null);
    const [chordResultPath, setChordResultPath] = useState(null);
    const [chordResultName, setChordResultName] = useState('');
    const [chordWaveform, setChordWaveform] = useState('sine');
    const [mixBlend, setMixBlend] = useState(1.0);
    const [mixPreviewUrl, setMixPreviewUrl] = useState(null);
    const [mixPreviewing, setMixPreviewing] = useState(false);
    const [chordBaseMidi, setChordBaseMidi] = useState(48);
    const chordAudioRef = useRef(null);
    const chordFileRef = useRef(null);

    const [error, setError] = useState(null);

    // Cleanup blob URLs on unmount / close
    useEffect(() => {
        if (!open) {
            setMidiAudioUrl(null);
            setChordAudioUrl(null);
            setChordResultUrl(null);
            setMidiFile(null);
            setChordAudioFile(null);
            setChordText('');
            setChordResultPath(null);
            setChordResultName('');
            setChordResultDuration(null);
            setMidiPath(null);
            setMidiDuration(null);
            setMidiName('');
            setMixBlend(1.0);
            if (mixPreviewUrl) URL.revokeObjectURL(mixPreviewUrl);
            setMixPreviewUrl(null);
            setMixPreviewing(false);
            setError(null);
        }
    }, [open]);

    // ---- MIDI handlers ----
    const onPickMidi = () => midiFileRef.current?.click();

    const onMidiFileChange = async (e) => {
        const f = e.target.files?.[0];
        e.target.value = '';
        if (!f) return;
        setMidiFile(f);
        setMidiName(f.name);
        setError(null);
        setMidiRendering(true);
        try {
            const form = new FormData();
            form.append('file', f);
            form.append('waveform', midiWaveform);
            form.append('transpose', String(midiTranspose));
            const r = await api.post('/api/audio/midi/render', form);
            const { path, name, duration_seconds } = r.data;
            setMidiPath(path);
            setMidiDuration(duration_seconds);
            setMidiName(name);
            const blobR = await api.get(`/api/media/${path}`, { responseType: 'blob' });
            const url = URL.createObjectURL(blobR.data);
            if (midiAudioUrl) URL.revokeObjectURL(midiAudioUrl);
            setMidiAudioUrl(url);
        } catch (err) {
            setError(err.response?.data?.error?.message || err.message || 'MIDI render failed');
        } finally {
            setMidiRendering(false);
        }
    };

    const reRenderMidi = async () => {
        if (!midiFile) return;
        setError(null);
        setMidiRendering(true);
        try {
            const form = new FormData();
            form.append('file', midiFile);
            form.append('waveform', midiWaveform);
            form.append('transpose', String(midiTranspose));
            const r = await api.post('/api/audio/midi/render', form);
            const { path, name, duration_seconds } = r.data;
            setMidiPath(path);
            setMidiDuration(duration_seconds);
            setMidiName(name);
            const blobR = await api.get(`/api/media/${path}`, { responseType: 'blob' });
            const url = URL.createObjectURL(blobR.data);
            if (midiAudioUrl) URL.revokeObjectURL(midiAudioUrl);
            setMidiAudioUrl(url);
        } catch (err) {
            setError(err.response?.data?.error?.message || err.message || 'Re-render failed');
        } finally {
            setMidiRendering(false);
        }
    };

    // ---- Chord handlers ----
    const onPickChordAudio = () => chordFileRef.current?.click();

    const onChordAudioChange = async (e) => {
        const f = e.target.files?.[0];
        e.target.value = '';
        if (!f) return;
        setChordAudioFile(f);
        setChordText('');
        setChordResultPath(null);
        setChordResultUrl(null);
        setError(null);
        const url = URL.createObjectURL(f);
        if (chordAudioUrl) URL.revokeObjectURL(chordAudioUrl);
        setChordAudioUrl(url);
    };

    const extractChords = async () => {
        if (!chordAudioFile) return;
        setError(null);
        setChordExtracting(true);
        try {
            const form = new FormData();
            form.append('file', chordAudioFile);
            const r = await api.post('/api/audio/chord-to-sine/extract', form);
            setChordText(r.data.chord_text);
        } catch (err) {
            setError(err.response?.data?.error?.message || err.message || 'Chord extraction failed');
        } finally {
            setChordExtracting(false);
        }
    };

    const renderChordSine = async () => {
        if (!chordText.trim()) {
            setError('No chord text to render. Extract chords first.');
            return;
        }
        setError(null);
        setChordRendering(true);
        try {
            const r = await api.post('/api/audio/chord-to-sine/render', {
                chord_text: chordText,
                waveform: chordWaveform,
                balance: 1.0,
                base_midi: chordBaseMidi,
            });
            const { path, name, duration_seconds } = r.data;
            setChordResultPath(path);
            setChordResultDuration(duration_seconds);
            setChordResultName(name);
            const blobR = await api.get(`/api/media/${path}`, { responseType: 'blob' });
            const url = URL.createObjectURL(blobR.data);
            if (chordResultUrl) URL.revokeObjectURL(chordResultUrl);
            setChordResultUrl(url);
        } catch (err) {
            setError(err.response?.data?.error?.message || err.message || 'Sine render failed');
        } finally {
            setChordRendering(false);
        }
    };

    // ---- Mix preview ----
    const generateMixPreview = async () => {
        if (!chordAudioUrl || !chordResultUrl) return;
        setMixPreviewing(true);
        setError(null);
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        try {
            const origResp = await fetch(chordAudioUrl);
            const origBlob = await origResp.blob();
            const sineResp = await fetch(chordResultUrl);
            const sineBlob = await sineResp.blob();
            const origBuf = await blobToAudioBuffer(ctx, origBlob);
            const sineBuf = await blobToAudioBuffer(ctx, sineBlob);
            const outLen = Math.max(origBuf.length, sineBuf.length);
            const sr = origBuf.sampleRate;
            const nCh = Math.max(origBuf.numberOfChannels, sineBuf.numberOfChannels);
            const outBuf = ctx.createBuffer(nCh, outLen, sr);
            for (let c = 0; c < nCh; c++) {
                const origChan = c < origBuf.numberOfChannels ? origBuf.getChannelData(c) : null;
                // Duplicate mono sine to all channels so stereo originals
                // don't end up with sine only on the left side.
                const sineIdx = Math.min(c, sineBuf.numberOfChannels - 1);
                const sineChan = sineBuf.numberOfChannels > 0 ? sineBuf.getChannelData(sineIdx) : null;
                const dst = outBuf.getChannelData(c);
                for (let i = 0; i < outLen; i++) {
                    const o = origChan ? origChan[Math.min(i, origBuf.length - 1)] : 0;
                    const s = sineChan ? sineChan[Math.min(i, sineBuf.length - 1)] : 0;
                    dst[i] = o * (1 - mixBlend) + s * mixBlend;
                }
            }
            const wavBlob = audioBufferToWav(outBuf);
            const url = URL.createObjectURL(wavBlob);
            if (mixPreviewUrl) URL.revokeObjectURL(mixPreviewUrl);
            setMixPreviewUrl(url);
        } catch (err) {
            setError('Mix preview failed: ' + (err.message || err));
        } finally {
            ctx.close();
            setMixPreviewing(false);
        }
    };

    // ---- Mix & apply ----
    const mixAndUpload = async () => {
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        try {
            // Fetch both blobs
            const origResp = await fetch(chordAudioUrl);
            const origBlob = await origResp.blob();
            const sineResp = await fetch(chordResultUrl);
            const sineBlob = await sineResp.blob();

            const origBuf = await blobToAudioBuffer(ctx, origBlob);
            const sineBuf = await blobToAudioBuffer(ctx, sineBlob);

            // Use the longer duration so nothing is clipped
            const outLen = Math.max(origBuf.length, sineBuf.length);
            const sr = origBuf.sampleRate;
            const nCh = Math.max(origBuf.numberOfChannels, sineBuf.numberOfChannels);
            const outBuf = ctx.createBuffer(nCh, outLen, sr);

            for (let c = 0; c < nCh; c++) {
                const origChan = c < origBuf.numberOfChannels ? origBuf.getChannelData(c) : null;
                // Duplicate mono sine to all channels so stereo originals
                // don't end up with sine only on the left side.
                const sineIdx = Math.min(c, sineBuf.numberOfChannels - 1);
                const sineChan = sineBuf.numberOfChannels > 0 ? sineBuf.getChannelData(sineIdx) : null;
                const dst = outBuf.getChannelData(c);
                for (let i = 0; i < outLen; i++) {
                    const o = origChan ? origChan[Math.min(i, origBuf.length - 1)] : 0;
                    const s = sineChan ? sineChan[Math.min(i, sineBuf.length - 1)] : 0;
                    dst[i] = o * (1 - mixBlend) + s * mixBlend;
                }
            }

            const wavBlob = audioBufferToWav(outBuf);
            const name = `mixed_${Date.now()}.wav`;
            const file = new File([wavBlob], name, { type: 'audio/wav' });
            const form = new FormData();
            form.append('file', file);
            const uploadResp = await api.post('/api/audio/upload', form);
            return uploadResp.data.path;
        } finally {
            ctx.close();
        }
    };

    const handleApply = async () => {
        if (tab === 0) {
            if (midiPath) {
                onApplySource(midiPath);
                onClose();
            }
            return;
        }
        // tab === 1: chord tab
        if (!chordResultPath) return;
        if (mixBlend < 1 && chordAudioUrl) {
            try {
                const mixedPath = await mixAndUpload();
                onApplySource(mixedPath);
                onClose();
            } catch (err) {
                setError('Mixing failed: ' + (err.message || err));
            }
        } else {
            onApplySource(chordResultPath);
            onClose();
        }
    };

    const canApply = (tab === 0 && !!midiPath) || (tab === 1 && !!chordResultPath);

    return (
        <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth scroll="paper">
            <DialogTitle>
                <Stack direction="row" alignItems="center" spacing={1}>
                    <MusicIcon size={20} />
                    <Typography variant="h6">Reference Audio Generator</Typography>
                </Stack>
            </DialogTitle>
            <DialogContent dividers>
                <Tabs value={tab} onChange={(_, v) => setTab(v)} sx={{ mb: 2 }}>
                    <Tab label="MIDI to Sine" />
                    <Tab label="Convert Audio to Chord Progression" />
                </Tabs>

                {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}

                {/* ---- TAB 0: MIDI ---- */}
                {tab === 0 && (
                    <Box>
                        {!midiFile ? (
                            <Button
                                variant="outlined"
                                startIcon={<UploadIcon size={14} />}
                                onClick={onPickMidi}
                                disabled={midiRendering}
                                fullWidth
                                sx={{ borderStyle: 'dashed', mb: 2 }}
                            >
                                {midiRendering ? 'Rendering MIDI…' : 'Choose a MIDI file…'}
                            </Button>
                        ) : (
                            <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 2, p: 1, border: '1px dashed', borderColor: 'divider', borderRadius: 1 }}>
                                <Typography variant="body2" sx={{ flex: 1, fontFamily: 'monospace', fontSize: 12, overflow: 'hidden', textOverflow: 'ellipsis' }}>
                                    {midiName}
                                    {midiDuration && ` · ${midiDuration.toFixed(2)}s`}
                                </Typography>
                                <IconButton size="small" onClick={() => { setMidiFile(null); setMidiAudioUrl(null); setMidiPath(null); setMidiDuration(null); }}>
                                    <ClearIcon size={14} />
                                </IconButton>
                            </Stack>
                        )}
                        <input ref={midiFileRef} type="file" accept=".mid,.midi" style={{ display: 'none' }} onChange={onMidiFileChange} />

                        <Stack spacing={2} sx={{ mb: 2 }}>
                            <Box>
                                <Typography variant="caption" color="text.secondary">Waveform</Typography>
                                <Stack direction="row" spacing={1} sx={{ mt: 0.5 }}>
                                    {['sine', 'triangle'].map(w => (
                                        <Button
                                            key={w}
                                            size="small"
                                            variant={midiWaveform === w ? 'contained' : 'outlined'}
                                            onClick={() => { setMidiWaveform(w); if (midiFile) reRenderMidi(); }}
                                            sx={{ textTransform: 'none' }}
                                        >
                                            {w}
                                        </Button>
                                    ))}
                                </Stack>
                            </Box>
                            <Box>
                                <Typography variant="caption" color="text.secondary">Transpose (±12 semitones)</Typography>
                                <Stack direction="row" alignItems="center" spacing={2}>
                                    <Slider
                                        value={midiTranspose}
                                        onChange={(_, v) => setMidiTranspose(v)}
                                        onChangeCommitted={() => { if (midiFile) reRenderMidi(); }}
                                        min={-12}
                                        max={12}
                                        step={1}
                                        valueLabelDisplay="auto"
                                        sx={{ flex: 1 }}
                                    />
                                    <Typography variant="body2" sx={{ width: 32, textAlign: 'right' }}>
                                        {midiTranspose > 0 ? '+' : ''}{midiTranspose}
                                    </Typography>
                                </Stack>
                            </Box>
                        </Stack>

                        {midiRendering && <LinearProgress sx={{ mb: 2 }} />}

                        {midiAudioUrl && (
                            <Box sx={{ mb: 2 }}>
                                <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 0.5 }}>
                                    Preview
                                </Typography>
                                <audio controls src={midiAudioUrl} style={{ width: '100%' }} />
                            </Box>
                        )}
                    </Box>
                )}

                {/* ---- TAB 1: Convert Audio to Chord Progression ---- */}
                {tab === 1 && (
                    <Box>
                        {!chordAudioFile ? (
                            <Button
                                variant="outlined"
                                startIcon={<UploadIcon size={14} />}
                                onClick={onPickChordAudio}
                                disabled={chordExtracting}
                                fullWidth
                                sx={{ borderStyle: 'dashed', mb: 2 }}
                            >
                                {chordExtracting ? 'Extracting chords…' : 'Upload audio for chord extraction…'}
                            </Button>
                        ) : (
                            <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 2, p: 1, border: '1px dashed', borderColor: 'divider', borderRadius: 1 }}>
                                <Typography variant="body2" sx={{ flex: 1, fontFamily: 'monospace', fontSize: 12, overflow: 'hidden', textOverflow: 'ellipsis' }}>
                                    {chordAudioFile.name}
                                </Typography>
                                <IconButton size="small" onClick={() => { setChordAudioFile(null); setChordAudioUrl(null); setChordText(''); setChordResultPath(null); setChordResultUrl(null); }}>
                                    <ClearIcon size={14} />
                                </IconButton>
                            </Stack>
                        )}
                        <input ref={chordFileRef} type="file" accept=".wav,.mp3,.flac,.m4a,.ogg,.opus,audio/*" style={{ display: 'none' }} onChange={onChordAudioChange} />

                        {chordAudioUrl && (
                            <Box sx={{ mb: 2 }}>
                                <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 0.5 }}>
                                    Original audio
                                </Typography>
                                <audio controls src={chordAudioUrl} style={{ width: '100%' }} />
                            </Box>
                        )}

                        <Button
                            variant="contained"
                            onClick={extractChords}
                            disabled={!chordAudioFile || chordExtracting}
                            fullWidth
                            sx={{ mb: 2 }}
                        >
                            {chordExtracting ? 'Extracting…' : 'Extract chords'}
                        </Button>

                        {chordExtracting && <LinearProgress sx={{ mb: 2 }} />}

                        <TextField
                            label="Chord text (editable)"
                            multiline
                            minRows={6}
                            maxRows={12}
                            value={chordText}
                            onChange={(e) => setChordText(e.target.value)}
                            placeholder={chordExtracting ? 'Extracting…' : '# start  end    chord\n0.0     4.0    C:maj\n4.0     8.0    G:maj\n…'}
                            fullWidth
                            sx={{ mb: 2 }}
                            InputProps={{ sx: { fontFamily: 'monospace', fontSize: 13 } }}
                        />

                        <Stack spacing={2} sx={{ mb: 2 }}>
                            <Box>
                                <Typography variant="caption" color="text.secondary">Waveform</Typography>
                                <Stack direction="row" spacing={1} sx={{ mt: 0.5 }}>
                                    {['sine', 'triangle'].map(w => (
                                        <Button
                                            key={w}
                                            size="small"
                                            variant={chordWaveform === w ? 'contained' : 'outlined'}
                                            onClick={() => setChordWaveform(w)}
                                            sx={{ textTransform: 'none' }}
                                        >
                                            {w}
                                        </Button>
                                    ))}
                                </Stack>
                            </Box>
                            <Box>
                                <Typography variant="caption" color="text.secondary">Base MIDI note</Typography>
                                <TextField
                                    size="small"
                                    type="number"
                                    value={chordBaseMidi}
                                    onChange={(e) => setChordBaseMidi(parseInt(e.target.value) || 48)}
                                    inputProps={{ min: 24, max: 84, step: 1 }}
                                    sx={{ width: 100, mt: 0.5 }}
                                />
                            </Box>
                        </Stack>

                        <Button
                            variant="contained"
                            onClick={renderChordSine}
                            disabled={!chordText.trim() || chordRendering}
                            fullWidth
                            sx={{ mb: 2 }}
                        >
                            {chordRendering ? 'Rendering…' : 'Regenerate from chords'}
                        </Button>

                        {chordRendering && <LinearProgress sx={{ mb: 2 }} />}

                        {chordResultUrl && (
                            <Box sx={{ mb: 2 }}>
                                <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 0.5 }}>
                                    Chord sine {chordResultDuration && `(${chordResultDuration.toFixed(2)}s)`}
                                </Typography>
                                <audio controls src={chordResultUrl} style={{ width: '100%' }} />
                            </Box>
                        )}

                        {/* Mix slider: blend original with chord sine */}
                        {chordAudioUrl && chordResultUrl && (
                            <Box sx={{ mb: 2 }}>
                                <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 0.5 }}>
                                    Mix: original ↔ chord sine ({mixBlend.toFixed(2)})
                                </Typography>
                                <Stack direction="row" alignItems="center" spacing={2} sx={{ mb: 1 }}>
                                    <Typography variant="caption" color="text.secondary" sx={{ minWidth: 48 }}>
                                        Original
                                    </Typography>
                                    <Slider
                                        value={mixBlend}
                                        onChange={(_, v) => { setMixBlend(v); if (mixPreviewUrl) { URL.revokeObjectURL(mixPreviewUrl); setMixPreviewUrl(null); } }}
                                        min={0}
                                        max={1}
                                        step={0.05}
                                        valueLabelDisplay="auto"
                                        sx={{ flex: 1 }}
                                    />
                                    <Typography variant="caption" color="text.secondary" sx={{ minWidth: 32 }}>
                                        Sine
                                    </Typography>
                                </Stack>
                                <Button
                                    variant="outlined"
                                    size="small"
                                    onClick={generateMixPreview}
                                    disabled={mixPreviewing}
                                    fullWidth
                                    sx={{ textTransform: 'none', mb: 1 }}
                                >
                                    {mixPreviewing ? 'Mixing…' : mixPreviewUrl ? 'Refresh mix preview' : 'Preview mix'}
                                </Button>
                                {mixPreviewUrl && (
                                    <Box>
                                        <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 0.5 }}>
                                            Mixed preview
                                        </Typography>
                                        <audio controls src={mixPreviewUrl} style={{ width: '100%' }} />
                                    </Box>
                                )}
                            </Box>
                        )}
                    </Box>
                )}
            </DialogContent>
            <DialogActions>
                <Button onClick={onClose}>Cancel</Button>
                <Button variant="contained" onClick={handleApply} disabled={!canApply}>
                    Use as source
                </Button>
            </DialogActions>
        </Dialog>
    );
}
