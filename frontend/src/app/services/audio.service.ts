import { Injectable, signal, OnDestroy } from '@angular/core';

@Injectable({
  providedIn: 'root'
})
export class AudioService implements OnDestroy {
  private audio = new Audio();
  
  // Expose state as Signals
  public isPlaying = signal<boolean>(false);
  public currentTime = signal<number>(0);
  public duration = signal<number>(0);
  public volume = signal<number>(0.8);
  public currentTrackPath = signal<string>('');

  private listeners: { name: string; handler: any }[] = [];

  constructor() {
    this.audio.volume = this.volume();
    
    // Register audio event listeners
    this.addListener('timeupdate', () => {
      this.currentTime.set(this.audio.currentTime);
    });

    this.addListener('loadedmetadata', () => {
      this.duration.set(this.audio.duration);
    });

    this.addListener('ended', () => {
      this.isPlaying.set(false);
      this.currentTime.set(0);
    });

    this.addListener('pause', () => {
      this.isPlaying.set(false);
    });

    this.addListener('play', () => {
      this.isPlaying.set(true);
    });
  }

  private addListener(event: string, handler: any) {
    this.audio.addEventListener(event, handler);
    this.listeners.push({ name: event, handler });
  }

  public loadTrack(path: string) {
    if (!path) return;
    this.currentTrackPath.set(path);
    this.audio.src = `http://127.0.0.1:8000/api/stream-audio?path=${encodeURIComponent(path)}`;
    this.audio.load();
    this.isPlaying.set(false);
    this.currentTime.set(0);
    this.duration.set(0);
  }

  public play() {
    if (!this.audio.src) return;
    this.audio.play().then(() => {
      this.isPlaying.set(true);
    }).catch(err => {
      console.error('Audio playback error:', err);
    });
  }

  public pause() {
    this.audio.pause();
    this.isPlaying.set(false);
  }

  public togglePlay() {
    if (this.isPlaying()) {
      this.pause();
    } else {
      this.play();
    }
  }

  public seek(time: number) {
    if (isNaN(time) || time < 0) return;
    this.audio.currentTime = time;
    this.currentTime.set(time);
  }

  public setVolume(vol: number) {
    const safeVol = Math.max(0, Math.min(1, vol));
    this.audio.volume = safeVol;
    this.volume.set(safeVol);
  }

  ngOnDestroy() {
    this.audio.pause();
    this.listeners.forEach(l => {
      this.audio.removeEventListener(l.name, l.handler);
    });
  }
}
