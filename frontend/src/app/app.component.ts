import { Component, OnInit, OnDestroy, effect } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { 
  LucideMusic, 
  LucideUpload, 
  LucideLayers, 
  LucideFileText, 
  LucidePlay, 
  LucidePause, 
  LucideVolume2, 
  LucideFolderOpen, 
  LucideAlertCircle, 
  LucideRefreshCw, 
  LucideClipboardCheck 
} from '@lucide/angular';
import { WaveformComponent } from './components/waveform/waveform.component';
import { ReviewEditorComponent } from './components/review-editor/review-editor.component';
import { LyricsSyncerComponent } from './components/lyrics-syncer/lyrics-syncer.component';
import { ChordSheetEditorComponent } from './components/chord-sheet-editor/chord-sheet-editor.component';
import { ApiService, BarData, ChordData, LyricsLine } from './services/api.service';
import { AudioService } from './services/audio.service';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    WaveformComponent,
    ReviewEditorComponent,
    LyricsSyncerComponent,
    ChordSheetEditorComponent,
    LucideMusic,
    LucideUpload,
    LucideLayers,
    LucideFileText,
    LucidePlay,
    LucidePause,
    LucideVolume2,
    LucideFolderOpen,
    LucideAlertCircle,
    LucideRefreshCw,
    LucideClipboardCheck
  ],
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit, OnDestroy {
  public activeTab: 'upload' | 'review' | 'sync' | 'editor' = 'upload';
  
  // Extraction states
  public audioPath = '';
  public filename = '';
  public chords: ChordData[] = [];
  public extractionStatus: 'idle' | 'extracting' | 'completed' | 'failed' = 'idle';
  public extractionProgress = 0;
  public extractionMessage = '';
  
  // Editor/Project state
  public chordsheetText = '';
  public timestamps: number[] = [];
  public unsyncedLyrics = '';
  public lyrics: LyricsLine[] = [];
  public bpm: number | null = null;
  public bars: BarData[] = [];
  public autoSynced = false;
  public estimatedLyricsStart = 0;
  public selectedStartBarTime = 0;
  public error = '';
  public success = '';
  public youtubeUrl = '';
  
  // Hover playbar state
  public showHoverTooltip = false;
  public hoverLeft = 0;
  public hoverTimeText = '';

  private pollingTimer: any = null;

  constructor(
    public audioService: AudioService,
    private apiService: ApiService
  ) {
    // Sync volume with signal updates
    effect(() => {
      this.audioService.setVolume(this.audioService.volume());
    });
  }

  ngOnInit() {
    // Initial load configurations
  }

  ngOnDestroy() {
    this.clearPolling();
  }

  public async handleSelectFile() {
    this.error = '';
    this.success = '';
    try {
      const data = await this.apiService.selectFile();
      if (data.status === 'selected') {
        this.audioPath = data.path;
        this.filename = data.filename;
        
        // Reset state
        this.audioService.loadTrack(data.path);
        this.chords = [];
        this.chordsheetText = '';
        this.timestamps = [];
        this.unsyncedLyrics = '';
        this.lyrics = [];
        this.bpm = null;
        this.bars = [];
        this.autoSynced = false;
        this.estimatedLyricsStart = 0;
        this.selectedStartBarTime = 0;
        
        // Trigger extraction
        this.startChordExtraction(data.path);
      }
    } catch (err) {
      this.error = 'Could not connect to the backend. Is python app.py running?';
    }
  }

  public async startChordExtraction(path: string) {
    this.extractionStatus = 'extracting';
    this.extractionProgress = 0;
    this.extractionMessage = 'Initializing...';

    try {
      const data = await this.apiService.extractChords(path);
      const taskId = data.task_id;
      
      this.clearPolling();
      this.pollingTimer = setInterval(() => {
        this.pollStatus(taskId);
      }, 1000);
    } catch (err: any) {
      this.extractionStatus = 'failed';
      this.extractionMessage = `Error starting extraction: ${err.message}`;
    }
  }

  public async handleSelectYoutube() {
    if (!this.youtubeUrl || !this.youtubeUrl.trim()) return;
    this.error = '';
    this.success = '';
    this.extractionStatus = 'extracting';
    this.extractionProgress = 0;
    this.extractionMessage = 'Connecting to YouTube...';
    
    // Reset state
    this.unsyncedLyrics = '';
    this.lyrics = [];
    this.bpm = null;
    this.bars = [];
    this.autoSynced = false;
    this.estimatedLyricsStart = 0;
    this.selectedStartBarTime = 0;

    try {
      const data = await this.apiService.extractYoutube(this.youtubeUrl);
      const taskId = data.task_id;

      this.clearPolling();
      this.pollingTimer = setInterval(() => {
        this.pollStatus(taskId);
      }, 1000);
    } catch (err: any) {
      this.extractionStatus = 'failed';
      this.extractionMessage = `Error: ${err.message}`;
    }
  }

  private async pollStatus(taskId: string) {
    try {
      const data = await this.apiService.getExtractionStatus(taskId);
      this.extractionProgress = data.progress;
      this.extractionMessage = data.message;

      if (data.status === 'completed') {
        const res = data.result;
        this.chords = res.chords;
        this.bpm = res.bpm;
        this.bars = res.bars;
        this.lyrics = res.lyrics || [];
        this.autoSynced = res.auto_synced || false;
        this.estimatedLyricsStart = res.estimatedLyricsStart || 0.0;
        this.extractionStatus = 'completed';
        
        if (res.audioPath) {
          this.audioPath = res.audioPath;
          this.audioService.loadTrack(res.audioPath);
        }
        if (res.filename) this.filename = res.filename;
        if (res.unsyncedLyrics) this.unsyncedLyrics = res.unsyncedLyrics;

        // Switch to Verify & Edit screen
        this.activeTab = 'review';
        this.success = 'Chords successfully extracted! Please review chords by bars and edit the lyrics before final syncing.';
        this.clearPolling();
      } else if (data.status === 'failed') {
        this.extractionStatus = 'failed';
        this.clearPolling();
      }
    } catch (err) {
      console.error('Error polling extraction status:', err);
    }
  }

  private clearPolling() {
    if (this.pollingTimer) {
      clearInterval(this.pollingTimer);
      this.pollingTimer = null;
    }
  }

  // Audio delegates
  public togglePlayPause() {
    this.audioService.togglePlay();
  }

  public handleSeek(time: number) {
    this.audioService.seek(time);
  }

  public formatTime(secs: number): string {
    if (!secs || isNaN(secs)) return '00:00';
    const m = Math.floor(secs / 60);
    const s = Math.floor(secs % 60);
    return `${m < 10 ? '0' : ''}${m}:${s < 10 ? '0' : ''}${s}`;
  }

  public handleSyncComplete(event: { chordsheet: string, timestamps: number[] }) {
    this.chordsheetText = event.chordsheet;
    this.timestamps = event.timestamps;
    this.activeTab = 'editor';
    this.success = 'Chord sheet generated! Customize spacing inside the editor.';
  }

  public get reviewLyricsText(): string {
    if (this.autoSynced && this.lyrics) {
      return this.lyrics.map(l => l.text).join('\n');
    }
    return this.unsyncedLyrics;
  }

  public handlePlayBarMouseMove(event: MouseEvent) {
    const container = event.currentTarget as HTMLElement;
    if (!container) return;
    const rect = container.getBoundingClientRect();
    const x = event.clientX - rect.left;
    this.hoverLeft = x;
    
    const ratio = Math.max(0, Math.min(1, x / rect.width));
    const duration = this.audioService.duration() || 0;
    const hoverTime = ratio * duration;
    
    this.hoverTimeText = this.formatTime(hoverTime);
  }

  private alignLyrics(lyricLines: string[], originalLyrics: LyricsLine[]): { text: string, time: number, duration: number }[] {
    const N = lyricLines.length;
    const M = originalLyrics.length;
    
    const dp: number[][] = Array(N + 1).fill(0).map(() => Array(M + 1).fill(0));
    const parent: [number, number][][] = Array(N + 1).fill(0).map(() => Array(M + 1).fill(0));

    const getSimilarity = (s1: string, s2: string): number => {
      const w1 = s1.toLowerCase().replace(/[^\w\s\u0590-\u05FF]/g, '').split(/\s+/).filter(Boolean);
      const w2 = s2.toLowerCase().replace(/[^\w\s\u0590-\u05FF]/g, '').split(/\s+/).filter(Boolean);
      if (w1.length === 0 && w2.length === 0) return 1.0;
      if (w1.length === 0 || w2.length === 0) return 0.0;
      
      let matches = 0;
      const set2 = new Set(w2);
      for (const w of w1) {
        if (set2.has(w)) matches++;
      }
      return (2 * matches) / (w1.length + w2.length);
    };

    for (let i = 1; i <= N; i++) {
      for (let j = 1; j <= M; j++) {
        const sim = getSimilarity(lyricLines[i - 1], originalLyrics[j - 1].text);
        
        const scoreMatch = dp[i - 1][j - 1] + sim;
        const scoreSkipB = dp[i][j - 1];
        const scoreSkipA = dp[i - 1][j];

        if (scoreMatch >= scoreSkipB && scoreMatch >= scoreSkipA) {
          dp[i][j] = scoreMatch;
          parent[i][j] = [i - 1, j - 1];
        } else if (scoreSkipB >= scoreSkipA) {
          dp[i][j] = scoreSkipB;
          parent[i][j] = [i, j - 1];
        } else {
          dp[i][j] = scoreSkipA;
          parent[i][j] = [i - 1, j];
        }
      }
    }

    const matches: { [key: number]: number } = {};
    let curI = N;
    let curJ = M;
    while (curI > 0 && curJ > 0) {
      const [prevI, prevJ] = parent[curI][curJ];
      if (prevI === curI - 1 && prevJ === curJ - 1) {
        matches[curI - 1] = curJ - 1;
      }
      curI = prevI;
      curJ = prevJ;
    }

    let lastTime = 0.0;
    return lyricLines.map((line, idx) => {
      const origIdx = matches[idx];
      const hasMatch = origIdx !== undefined;
      const original = hasMatch ? originalLyrics[origIdx] : null;
      
      let origTime = 0.0;
      if (original) {
        origTime = original.time;
      } else {
        origTime = lastTime + 4.0;
      }
      lastTime = origTime;
      
      return {
        text: line,
        time: origTime,
        duration: original ? (original.duration || 0.0) : 0.0
      };
    });
  }

  public async handleReviewApprove(event: { chords: ChordData[], compiledBars: BarData[], localLyrics: string, selectedBarTime: number }) {
    this.chords = event.chords;
    this.bars = event.compiledBars;
    this.selectedStartBarTime = event.selectedBarTime;

    const lyricLines = event.localLyrics.split('\n').map(l => l.trim()).filter(l => l.length > 0);

    if (this.autoSynced && this.lyrics.length > 0) {
      // Align edited lines with original timestamps using dynamic sequence alignment
      const alignedBase = this.alignLyrics(lyricLines, this.lyrics);
      
      // Calculate offset shift relative to the new first line's original time
      const originalFirstTime = alignedBase[0] ? alignedBase[0].time : (this.lyrics[0] ? this.lyrics[0].time : 0);
      const offset = event.selectedBarTime - originalFirstTime;

      const alignedLyrics = alignedBase.map(item => {
        return {
          text: item.text,
          time: Math.max(0.0, item.time + offset),
          duration: item.duration
        };
      });

      this.error = '';
      this.success = '';
      try {
        const result = await this.apiService.generateChordsheet(
          event.chords,
          alignedLyrics,
          this.audioService.duration() || 300.0
        );
        this.chordsheetText = result.chordsheet;
        this.timestamps = result.timestamps;
        this.activeTab = 'editor';
        this.success = 'Chord sheet synced successfully! Alignment shifted by selected bar start.';
      } catch (err: any) {
        this.error = err.message || 'Server error occurred during chordsheet synchronization.';
      }
    } else {
      // Manual sync flow
      this.unsyncedLyrics = event.localLyrics;
      this.activeTab = 'sync';

      if (event.selectedBarTime > 0) {
        const seekTarget = Math.max(0.0, event.selectedBarTime - 4.0);
        this.handleSeek(seekTarget);
      }
      this.success = `Lyrics and chords verified. Skip intro active: player seeked to ${Math.max(0, event.selectedBarTime - 4.0).toFixed(0)}s. Tap to sync!`;
    }
  }

  public async handleSaveProject() {
    this.error = '';
    this.success = '';
    try {
      const data = await this.apiService.saveProject(
        this.audioPath,
        this.chordsheetText,
        this.timestamps,
        this.bpm,
        this.bars
      );
      if (data.status === 'saved') {
        this.success = `Project saved to: ${data.path}`;
      }
    } catch (err) {
      this.error = 'Failed to save project.';
    }
  }

  public async handleLoadProject() {
    this.error = '';
    this.success = '';
    try {
      const data = await this.apiService.loadProject();
      if (data.status === 'loaded') {
        const proj = data.data;
        this.audioPath = proj.audioPath;
        this.filename = proj.audioPath.split(/[\\/]/).pop() || 'Loaded Audio';
        this.chordsheetText = proj.chordsheetText;
        this.timestamps = proj.timestamps;
        this.bpm = proj.bpm || null;
        this.bars = proj.bars || [];
        this.chords = []; // Loaded projects have sheet precompiled
        this.extractionStatus = 'completed';

        this.audioService.loadTrack(proj.audioPath);
        this.activeTab = 'editor';
        this.success = `Loaded project: ${data.path}`;
      }
    } catch (err) {
      this.error = 'Failed to load project.';
    }
  }
}
