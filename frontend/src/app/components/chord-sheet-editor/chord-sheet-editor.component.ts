import { Component, Input, Output, EventEmitter, OnInit, OnChanges, SimpleChanges, ElementRef, ViewChildren, QueryList, AfterViewChecked } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { LucideSave, LucideFileDown, LucideInfo, LucideHelpCircle } from '@lucide/angular';
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
  imports: [CommonModule, FormsModule, LucideSave, LucideFileDown, LucideInfo, LucideHelpCircle],
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
  @Output() onSave = new EventEmitter<void>();

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
    this.chordsheetText = value;
    this.chordsheetTextChange.emit(value);
    this.blocks = this.parseBlocks();
    this.updateActiveBlockIndex();
  }

  public hasHebrew(text: string): boolean {
    return /[\u0590-\u05FF]/.test(text || '');
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
