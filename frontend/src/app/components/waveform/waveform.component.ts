import { Component, Input, Output, EventEmitter, ElementRef, ViewChild, AfterViewInit, OnChanges, SimpleChanges } from '@angular/core';

@Component({
  selector: 'app-waveform',
  standalone: true,
  template: `
    <div style="position: relative; width: 100%; height: 100%; cursor: pointer;">
      <canvas
        #canvas
        (click)="handleCanvasClick($event)"
        style="width: 100%; height: 100%; display: block;"
      ></canvas>
    </div>
  `
})
export class WaveformComponent implements AfterViewInit, OnChanges {
  @Input() duration = 0;
  @Input() currentTime = 0;
  @Output() onSeek = new EventEmitter<number>();

  @ViewChild('canvas') canvasRef!: ElementRef<HTMLCanvasElement>;

  private barHeights: number[] = [];

  constructor() {
    this.barHeights = this.generateWaveform(120);
  }

  ngAfterViewInit() {
    this.draw();
  }

  ngOnChanges(changes: SimpleChanges) {
    if (changes['currentTime'] || changes['duration']) {
      this.draw();
    }
  }

  private generateWaveform(count: number): number[] {
    let seed = 42;
    const random = () => {
      const x = Math.sin(seed++) * 10000;
      return x - Math.floor(x);
    };

    const bars: number[] = [];
    for (let i = 0; i < count; i++) {
      const base = 0.2 + 0.6 * Math.sin((i / count) * Math.PI);
      const noise = (random() - 0.5) * 0.35;
      const height = Math.max(0.1, Math.min(0.95, base + noise));
      bars.push(height);
    }
    return bars;
  }

  private draw() {
    if (!this.canvasRef) return;
    const canvas = this.canvasRef.nativeElement;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.offsetWidth;
    const height = canvas.offsetHeight;
    if (width === 0 || height === 0) return;

    // Retina support
    const dpr = window.devicePixelRatio || 1;
    if (canvas.width !== width * dpr || canvas.height !== height * dpr) {
      canvas.width = width * dpr;
      canvas.height = height * dpr;
      ctx.scale(dpr, dpr);
    } else {
      ctx.clearRect(0, 0, width, height);
    }

    const progress = this.duration > 0 ? this.currentTime / this.duration : 0;
    const barWidth = width / this.barHeights.length;
    const padding = 2; // px spacing

    this.barHeights.forEach((val, idx) => {
      const x = idx * barWidth;
      const barH = val * (height - 8);
      const y = (height - barH) / 2;
      const isPassed = (idx / this.barHeights.length) <= progress;

      ctx.beginPath();
      // Use standard roundRect (modern standard browser support)
      if (typeof ctx.roundRect === 'function') {
        ctx.roundRect(x + padding / 2, y, barWidth - padding, barH, 2);
      } else {
        ctx.rect(x + padding / 2, y, barWidth - padding, barH);
      }

      if (isPassed) {
        const gradient = ctx.createLinearGradient(0, y, 0, y + barH);
        gradient.addColorStop(0, '#818cf8'); // Indigo-400
        gradient.addColorStop(1, '#a855f7'); // Purple-500
        ctx.fillStyle = gradient;
      } else {
        ctx.fillStyle = 'rgba(255, 255, 255, 0.12)';
      }
      ctx.fill();
    });
  }

  handleCanvasClick(e: MouseEvent) {
    if (this.duration <= 0) return;
    const canvas = this.canvasRef.nativeElement;
    const rect = canvas.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const ratio = clickX / rect.width;
    const newTime = ratio * this.duration;
    this.onSeek.emit(newTime);
  }
}
