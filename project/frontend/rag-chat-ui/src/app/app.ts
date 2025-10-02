import { Component, signal } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { ChatListComponent } from './components/chat-list/chat-list';
import { CommonModule } from '@angular/common';
import { ChatWindowComponent } from './components/chat-window/chat-window';
import { FileUploadComponent } from './components/file-upload/file-upload';

@Component({
  selector: 'app-root',
  imports: [CommonModule, ChatWindowComponent, FileUploadComponent], //,RouterOutlet
  
  templateUrl: './app.html',
  styleUrl: './app.scss'
})
export class App {
  protected readonly title = signal('rag-chat-ui');
}
