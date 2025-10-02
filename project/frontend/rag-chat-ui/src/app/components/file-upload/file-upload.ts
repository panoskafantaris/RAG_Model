import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ChatService } from '../../services/chat.service';
import { FileUploadModule } from 'primeng/fileupload';

@Component({
  selector: 'app-file-upload',
  standalone: true,
  imports: [CommonModule, FileUploadModule],
  templateUrl: './file-upload.html',
  styleUrl: './file-upload.scss'
})
export class FileUploadComponent {
  constructor(private chatService: ChatService) {}

  onUpload(event: any) {
    for (let file of event.files) {
      this.chatService.uploadFile(file).subscribe(res => {
        console.log("File uploaded:", res);
      });
    }
  }
}
