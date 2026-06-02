import { Component, Input, Output, EventEmitter, OnInit, OnChanges, SimpleChanges, ElementRef, ViewChild, HostListener } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { 
  LucidePlay, 
  LucidePause, 
  LucideRotateCcw, 
  LucideCheck, 
  LucideSparkles, 
  LucideAlertCircle, 
  LucideUndo2, 
  LucideAlignLeft 
} from '@lucide/angular';
import { ApiService, ChordData, LyricsLine } from '../../services/api.service';

interface SyncLine {
  text: string;
  time: number | null;
}

@Component({
  selector: 'app-lyrics-syncer',
  standalone: true,
  imports: [
    CommonModule, 
    FormsModule, 
    LucidePlay, 
    LucidePause, 
    LucideRotateCcw, 
    LucideCheck, 
    LucideSparkles, 
    LucideAlertCircle, 
    LucideUndo2, 
    LucideAlignLeft
  ],
  templateUrl: './lyrics-syncer.component.html',
  styleUrls: []
})
export class LyricsSyncerComponent implements OnInit, OnChanges {
  @Input() audioPath = '';
  @Input() isPlaying = false;
  @Input() currentTime = 0;
  @Input() duration = 0;
  @Input() chords: ChordData[] = [];
  @Input() rawLyrics = '';
  @Input() selectedStartBarTime = 0;

  @Output() onPlayPause = new EventEmitter<void>();
  @Output() onSyncComplete = new EventEmitter<{ chordsheet: string, timestamps: number[] }>();
  @Output() rawLyricsChange = new EventEmitter<string>();

  @ViewChild('timelineEndRef') timelineEndEl!: ElementRef<HTMLDivElement>;

  public syncMode = false;
  public lines: SyncLine[] = [];
  public currentIndex = 0;
  public isGenerating = false;
  public error = '';

  constructor(private apiService: ApiService) {}

  ngOnInit() {
    this.resetSyncState();
  }

  ngOnChanges(changes: SimpleChanges) {
    if (changes['rawLyrics']) {
      this.resetSyncState();
    }
  }

  private resetSyncState() {
    this.syncMode = false;
    this.lines = [];
    this.currentIndex = 0;
    this.error = '';
  }

  @HostListener('window:keydown', ['$event'])
  handleKeyDown(event: KeyboardEvent) {
    if (!this.syncMode || this.currentIndex >= this.lines.length || !this.isPlaying) return;

    if (event.key === ' ' || event.key === 'Spacebar' || event.code === 'Space' || event.key === 'Enter') {
      event.preventDefault(); // Stop scrolling
      this.markLineTimestamp();
    }
  }

  public hasHebrew(text: string): boolean {
    return /[\u0590-\u05FF]/.test(text || '');
  }

  public startSyncMode() {
    if (!this.rawLyrics || !this.rawLyrics.trim()) {
      this.error = 'Please paste or type some lyrics first.';
      return;
    }
    this.error = '';

    const parsedLines = this.rawLyrics
      .split('\n')
      .map(line => line.trim())
      .filter(line => line.length > 0)
      .map(text => ({ text, time: null }));

    if (parsedLines.length === 0) {
      this.error = 'Could not find any readable lyrics lines.';
      return;
    }

    this.lines = parsedLines;
    this.currentIndex = 0;
    this.syncMode = true;
  }

  public markLineTimestamp() {
    if (this.currentIndex >= this.lines.length) return;

    this.lines[this.currentIndex].time = this.currentTime;
    this.currentIndex++;

    // Scroll into view
    setTimeout(() => {
      this.scrollTimelineToBottom();
    }, 50);
  }

  public undoLastLine() {
    if (this.currentIndex === 0) return;
    this.currentIndex--;
    this.lines[this.currentIndex].time = null;
  }

  public resetSync() {
    this.lines = this.lines.map(line => ({ ...line, time: null }));
    this.currentIndex = 0;
  }

  public exitSyncMode() {
    this.syncMode = false;
    this.resetSync();
  }

  public handleLyricsTextChange(value: string) {
    this.rawLyrics = value;
    this.rawLyricsChange.emit(value);
  }

  public async handleGenerateSheet() {
    const synced = this.lines.filter(l => l.time !== null) as { text: string; time: number }[];
    if (synced.length === 0) {
      this.error = 'Please sync at least one line of lyrics before generating.';
      return;
    }

    this.isGenerating = true;
    this.error = '';

    // Shift timestamps so first tapped line matches selectedStartBarTime
    let finalSyncedLyrics: LyricsLine[] = synced.map(l => ({ text: l.text, time: l.time }));
    if (this.selectedStartBarTime > 0 && finalSyncedLyrics.length > 0) {
      const firstTappedTime = finalSyncedLyrics[0].time;
      const offset = this.selectedStartBarTime - firstTappedTime;
      finalSyncedLyrics = finalSyncedLyrics.map(l => ({
        text: l.text,
        time: Math.max(0.0, l.time + offset)
      }));
    }

    try {
      const result = await this.apiService.generateChordsheet(
        this.chords,
        finalSyncedLyrics,
        this.duration || 300.0
      );
      this.onSyncComplete.emit({
        chordsheet: result.chordsheet,
        timestamps: result.timestamps
      });
    } catch (err: any) {
      this.error = err.message || 'Server error occurred during chord sheet compilation.';
    } finally {
      this.isGenerating = false;
    }
  }

  public formatTime(secs: number | null): string {
    if (secs === null || secs === undefined) return '--:--';
    const m = Math.floor(secs / 60);
    const s = Math.floor(secs % 60);
    const ms = Math.floor((secs % 1) * 10);
    return `${m}:${s < 10 ? '0' : ''}${s}.${ms}`;
  }

  public hasSyncedLines(): boolean {
    return this.lines.some(l => l.time !== null);
  }

  private scrollTimelineToBottom() {
    if (this.timelineEndEl) {
      this.timelineEndEl.nativeElement.scrollIntoView({ behavior: 'smooth' });
    }
  }
}
