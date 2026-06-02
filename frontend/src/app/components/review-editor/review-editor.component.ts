import { Component, Input, Output, EventEmitter, OnInit, OnChanges, SimpleChanges, ElementRef, ViewChildren, QueryList, AfterViewChecked } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { LucideCheck, LucideInfo, LucideMusic, LucideActivity } from '@lucide/angular';
import { BarData, ChordData } from '../../services/api.service';

@Component({
  selector: 'app-review-editor',
  standalone: true,
  imports: [CommonModule, FormsModule, LucideCheck, LucideInfo, LucideMusic, LucideActivity],
  templateUrl: './review-editor.component.html',
  styleUrls: []
})
export class ReviewEditorComponent implements OnInit, OnChanges, AfterViewChecked {
  @Input() bpm: number | null = null;
  @Input() bars: BarData[] = [];
  @Input() lyricsText = '';
  @Input() duration = 0;
  @Input() autoSynced = false;
  @Input() estimatedLyricsStart = 0;
  @Input() currentTime = 0;

  @Output() onApprove = new EventEmitter<{ chords: ChordData[], compiledBars: BarData[], localLyrics: string, selectedBarTime: number }>();
  @Output() onSeek = new EventEmitter<number>();

  @ViewChildren('barCard') barCards!: QueryList<ElementRef<HTMLDivElement>>;

  public localBars: BarData[] = [];
  public localLyrics = '';
  public startBarIdx = 0;
  public activeBarIndex = -1;
  public error = '';
  
  private activeBarIndexChanged = false;

  constructor() {}

  ngOnInit() {
    this.localBars = JSON.parse(JSON.stringify(this.bars || []));
    this.localLyrics = this.lyricsText || '';
    this.startBarIdx = this.findClosestBarIdx();
  }

  ngOnChanges(changes: SimpleChanges) {
    if (changes['bars'] && changes['bars'].currentValue) {
      this.localBars = JSON.parse(JSON.stringify(changes['bars'].currentValue));
      this.startBarIdx = this.findClosestBarIdx();
    }
    if (changes['lyricsText']) {
      this.localLyrics = changes['lyricsText'].currentValue || '';
    }
    if (changes['estimatedLyricsStart'] || changes['bars']) {
      this.startBarIdx = this.findClosestBarIdx();
    }
    if (changes['currentTime']) {
      this.updateActiveBarIndex();
    }
  }

  ngAfterViewChecked() {
    if (this.activeBarIndexChanged) {
      this.scrollActiveBarIntoView();
      this.activeBarIndexChanged = false;
    }
  }

  private findClosestBarIdx(): number {
    if (!this.localBars || this.localBars.length === 0) return 0;
    let closestIdx = 0;
    let minDiff = Infinity;
    const target = this.estimatedLyricsStart || 0;
    for (let i = 0; i < this.localBars.length; i++) {
      const diff = Math.abs(this.localBars[i].time - target);
      if (diff < minDiff) {
        minDiff = diff;
        closestIdx = i;
      }
    }
    return closestIdx;
  }

  private updateActiveBarIndex() {
    if (!this.localBars || this.currentTime === undefined) return;
    let activeIdx = -1;
    for (let i = 0; i < this.localBars.length; i++) {
      if (this.localBars[i].time <= this.currentTime) {
        activeIdx = i;
      } else {
        break;
      }
    }
    if (activeIdx !== this.activeBarIndex) {
      this.activeBarIndex = activeIdx;
      this.activeBarIndexChanged = true;
    }
  }

  private scrollActiveBarIntoView() {
    if (this.barCards && this.activeBarIndex >= 0) {
      const cardsArray = this.barCards.toArray();
      const activeCard = cardsArray[this.activeBarIndex];
      if (activeCard) {
        activeCard.nativeElement.scrollIntoView({
          behavior: 'smooth',
          block: 'nearest'
        });
      }
    }
  }

  public hasHebrew(text: string): boolean {
    return /[\u0590-\u05FF]/.test(text || '');
  }

  public handleChordChange(barIdx: number, beatIdx: number, event: Event) {
    const target = event.target as HTMLInputElement;
    if (this.localBars[barIdx]) {
      this.localBars[barIdx].chords[beatIdx] = target.value;
    }
  }

  public getDisplayValue(barIdx: number, beatIdx: number): string {
    if (!this.localBars[barIdx]) return '';
    const val = this.localBars[barIdx].chords[beatIdx];
    if (val === "" || val === undefined) {
      return "";
    }

    // Find the previous non-empty chord in the grid
    let prevChord = "";
    let found = false;
    for (let i = barIdx; i >= 0; i--) {
      const startBeat = (i === barIdx) ? beatIdx - 1 : 3;
      for (let k = startBeat; k >= 0; k--) {
        const c = this.localBars[i].chords[k];
        if (c !== "" && c !== undefined) {
          prevChord = c;
          found = true;
          break;
        }
      }
      if (found) break;
    }

    if (val === prevChord) {
      return ""; // Hide duplicate chord
    }
    return val;
  }

  public getInheritedChord(barIdx: number, beatIdx: number): string {
    let prevChord = "";
    let found = false;
    for (let i = barIdx; i >= 0; i--) {
      const startBeat = (i === barIdx) ? beatIdx - 1 : 3;
      for (let k = startBeat; k >= 0; k--) {
        const c = this.localBars[i].chords[k];
        if (c !== "" && c !== undefined) {
          prevChord = c;
          found = true;
          break;
        }
      }
      if (found) break;
    }
    return prevChord;
  }

  private getChordsFromBars(): ChordData[] {
    const flatChords: ChordData[] = [];
    let currentChord = "";

    for (let i = 0; i < this.localBars.length; i++) {
      const bar = this.localBars[i];
      const nextBar = this.localBars[i + 1];
      const barTime = bar.time;
      const nextTime = nextBar ? nextBar.time : (this.duration || (barTime + 4.0));
      const beatDuration = (nextTime - barTime) / 4;

      for (let k = 0; k < 4; k++) {
        let chord = bar.chords[k];
        if (chord === "" || chord === undefined) {
          chord = currentChord;
        }

        const beatTime = barTime + k * beatDuration;

        if (chord !== currentChord) {
          flatChords.push({
            time: beatTime,
            chord: chord
          });
          currentChord = chord;
        }
      }
    }
    return flatChords;
  }

  public handleApproveClick() {
    const chords = this.getChordsFromBars();
    const selectedBarTime = this.localBars[this.startBarIdx] ? this.localBars[this.startBarIdx].time : 0;
    
    // Map empty strings in localBars to their inherited values
    let currentChord = "";
    const compiledBars = this.localBars.map(bar => {
      const filledChords = bar.chords.map(c => {
        if (c !== "" && c !== undefined) {
          currentChord = c;
        }
        return currentChord;
      });
      return {
        ...bar,
        chords: filledChords
      };
    });

    this.onApprove.emit({
      chords,
      compiledBars,
      localLyrics: this.localLyrics,
      selectedBarTime
    });
  }

  public handleBarClick(barTime: number) {
    this.onSeek.emit(barTime);
  }

  public selectStartBar(barIdx: number, event: Event) {
    event.stopPropagation(); // Avoid seeking
    this.startBarIdx = barIdx;
  }
}
