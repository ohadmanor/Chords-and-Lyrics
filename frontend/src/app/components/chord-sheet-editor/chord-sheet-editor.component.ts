import { Component, Input, Output, EventEmitter, OnInit, OnChanges, SimpleChanges, ElementRef, ViewChildren, QueryList, AfterViewChecked } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { LucideFileDown, LucideInfo, LucideHelpCircle } from '@lucide/angular';
import { BarData } from '../../services/api.service';

interface ChordSheetBlock {
  chordLine: string;
  lyricLine: string;
  time: number;
  index: number;
}

@Component({
  selector: 'app-chord-sheet-editor',
  standalone: true,
  imports: [CommonModule, FormsModule, LucideFileDown, LucideInfo, LucideHelpCircle],
  templateUrl: './chord-sheet-editor.component.html',
  styleUrls: []
})
export class ChordSheetEditorComponent implements OnInit, OnChanges, AfterViewChecked {
  @Input() chordsheetText = '';
  @Input() timestamps: number[] = [];
  @Input() currentTime = 0;
  @Input() bpm: number | null = null;
  @Input() bars: BarData[] = [];

  @Output() chordsheetTextChange = new EventEmitter<string>();
  @Output() onSeek = new EventEmitter<number>();

  @ViewChildren('barCard') barCards!: QueryList<ElementRef<HTMLDivElement>>;
  @ViewChildren('blockCard') blockCards!: QueryList<ElementRef<HTMLDivElement>>;

  public viewMode: 'sheet' | 'bars' = 'sheet';
  public activeBarIndex = -1;
  public activeBlockIndex = -1;
  public blocks: ChordSheetBlock[] = [];

  private activeBarIndexChanged = false;
  private activeBlockIndexChanged = false;

  constructor() {}

  ngOnInit() {
    this.blocks = this.parseBlocks();
  }

  ngOnChanges(changes: SimpleChanges) {
    if (changes['chordsheetText'] || changes['timestamps']) {
      this.blocks = this.parseBlocks();
      this.updateActiveBlockIndex();
    }
    if (changes['currentTime']) {
      this.updateActiveBarIndex();
      this.updateActiveBlockIndex();
    }
  }

  ngAfterViewChecked() {
    if (this.viewMode === 'bars' && this.activeBarIndexChanged) {
      this.scrollActiveBarIntoView();
      this.activeBarIndexChanged = false;
    }
    if (this.viewMode === 'sheet' && this.activeBlockIndexChanged) {
      this.scrollActiveBlockIntoView();
      this.activeBlockIndexChanged = false;
    }
  }

  public handleTextChange(value: string) {
    // The textarea shows a mirrored view for Hebrew songs (chord columns
    // flipped to match the Interactive Viewer). Convert back to the canonical
    // (logical) representation before storing/emitting. mirrorSheet is
    // self-inverse, so applying it again restores the original layout.
    const canonical = this.hasHebrew(value) ? this.mirrorSheet(value) : value;
    this.chordsheetText = canonical;
    this.chordsheetTextChange.emit(canonical);
    this.blocks = this.parseBlocks();
    this.updateActiveBlockIndex();
  }

  public hasHebrew(text: string): boolean {
    return /[\u0590-\u05FF]/.test(text || '');
  }

  public get isHebrewSong(): boolean {
    return this.hasHebrew(this.chordsheetText);
  }

  /**
   * Text shown in the monospace editor. For Hebrew songs the chord lines are
   * mirrored so the editor matches the Interactive Viewer (first chord on the
   * right above the first sung syllable). English songs are shown as-is.
   */
  public get displayChordsheetText(): string {
    return this.isHebrewSong ? this.mirrorSheet(this.chordsheetText) : this.chordsheetText;
  }

  /** A bars/instrumental line (e.g. "// C / G //") has no paired lyric line. */
  public isBarsLine(block: ChordSheetBlock): boolean {
    return !block.lyricLine || !block.lyricLine.trim();
  }

  /**
   * Chord line shown for a block. English songs and bars-only lines are
   * returned unchanged (rendered LTR). For Hebrew songs the chord columns are
   * mirrored so the first chord (earliest time, logical column 0) lands above
   * the first sung syllable, which sits on the right for RTL lyrics. Each chord
   * token keeps its readable left-to-right spelling.
   */
  public displayChordLine(block: ChordSheetBlock): string {
    if (!this.isHebrewSong || this.isBarsLine(block)) {
      return block.chordLine;
    }
    return this.mirrorChordLine(block.chordLine, block.lyricLine);
  }

  /**
   * Mirror every chord line in a full chord sheet relative to its paired lyric
   * line, using the same block pairing as the viewer. Bars-only/instrumental
   * lines (no lyric) and blank separators are left untouched. This function is
   * its own inverse, so it converts both canonical->display and display->canonical.
   */
  private mirrorSheet(text: string): string {
    if (!text) return text;
    const lines = text.split('\n');
    const out: string[] = [];
    let i = 0;
    while (i < lines.length) {
      if (!lines[i].trim()) {
        out.push(lines[i]);
        i++;
        continue;
      }
      const chordLine = lines[i];
      const lyricLine = (i + 1 < lines.length) ? lines[i + 1] : '';
      if (!lyricLine.trim()) {
        // Bars/instrumental line: no paired lyric, keep as-is.
        out.push(chordLine);
        i++;
      } else {
        out.push(this.mirrorChordLine(chordLine, lyricLine));
        out.push(lyricLine);
        i += 2;
      }
    }
    return out.join('\n');
  }

  private mirrorChordLine(chordLine: string, lyricLine: string): string {
    const width = Math.max(chordLine.length, lyricLine.length);
    if (width === 0) return chordLine;

    const cells: string[] = new Array(width).fill(' ');
    const tokenRegex = /\S+/g;
    let match: RegExpExecArray | null;
    while ((match = tokenRegex.exec(chordLine)) !== null) {
      const token = match[0];
      // Mirror the token so its right edge sits above the same lyric column.
      let start = width - match.index - token.length;
      if (start < 0) start = 0;
      for (let k = 0; k < token.length; k++) {
        const idx = start + k;
        if (idx >= 0 && idx < width) {
          cells[idx] = token[k];
        }
      }
    }
    return cells.join('');
  }

  private parseBlocks(): ChordSheetBlock[] {
    if (!this.chordsheetText) return [];

    const lines = this.chordsheetText.split('\n');
    const parsed: ChordSheetBlock[] = [];
    let timeIdx = 0;
    let i = 0;

    while (i < lines.length) {
      const line = lines[i];

      if (!line.trim()) {
        i++;
        continue;
      }

      const chordLine = line;
      const lyricLine = (i + 1 < lines.length) ? lines[i + 1] : "";

      const blockTime = this.timestamps[timeIdx] !== undefined 
        ? this.timestamps[timeIdx] 
        : (timeIdx > 0 ? this.timestamps[timeIdx - 1] + 4.0 : 0.0);

      parsed.push({
        chordLine,
        lyricLine,
        time: blockTime,
        index: timeIdx
      });

      timeIdx++;
      i += 2;

      // Skip blank line separator
      if (i < lines.length && !lines[i].trim()) {
        i++;
      }
    }
    return parsed;
  }

  private updateActiveBarIndex() {
    if (!this.bars) return;
    let activeIdx = -1;
    for (let i = 0; i < this.bars.length; i++) {
      if (this.bars[i].time <= this.currentTime) {
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

  private updateActiveBlockIndex() {
    let activeIdx = -1;
    for (let j = 0; j < this.blocks.length; j++) {
      if (this.blocks[j].time <= this.currentTime) {
        activeIdx = j;
      } else {
        break;
      }
    }
    if (activeIdx !== this.activeBlockIndex) {
      this.activeBlockIndex = activeIdx;
      this.activeBlockIndexChanged = true;
    }
  }

  private scrollActiveBarIntoView() {
    if (this.barCards && this.activeBarIndex >= 0) {
      const cardsArray = this.barCards.toArray();
      const activeCard = cardsArray[this.activeBarIndex];
      if (activeCard) {
        activeCard.nativeElement.scrollIntoView({
          behavior: 'smooth',
          block: 'center'
        });
      }
    }
  }

  private scrollActiveBlockIntoView() {
    if (this.blockCards && this.activeBlockIndex >= 0) {
      const blocksArray = this.blockCards.toArray();
      const activeBlock = blocksArray[this.activeBlockIndex];
      if (activeBlock) {
        activeBlock.nativeElement.scrollIntoView({
          behavior: 'smooth',
          block: 'center'
        });
      }
    }
  }

  public handleSeek(time: number) {
    this.onSeek.emit(time);
  }

  public handleExportText() {
    const element = document.createElement("a");
    const file = new Blob([this.chordsheetText], { type: 'text/plain' });
    element.href = URL.createObjectURL(file);
    element.download = "song_sheet.txt";
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  }
}
